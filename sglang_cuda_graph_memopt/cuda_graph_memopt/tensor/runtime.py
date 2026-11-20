"""Tensor placement at capture time: record -> plan -> packed capture.

Requires the patched ``libc10_cuda.so`` (``cuda_graph_memopt/build.sh c10``), which adds
the ``tmo_*`` C symbols to PyTorch's caching allocator:

    tmo_trace_begin/end/count/copy_trace   log of private-pool alloc / free_completed events
    tmo_set_capture_plan[_at] / clear / cursor
                                           serve the n-th malloc of a capture from a planned offset
                                           (_at: inside the arena an earlier capture reserved)
    tmo_capture_plan_arena_base            where the arena of the last plan lives
    tmo_va_only_begin/end, tmo_vmm_commit  reserve the arena as VMM address space, map physical
    tmo_vmm_physical_bytes                 pages only after the plan exists

Flow (details in cuda-graph-memopt-context.md, section "Tensor planning"):

    1. record pass : fn() runs inside a throw-away CUDAGraph and a private pool. The allocator
                     logs every alloc / free_completed. The graph is dropped afterwards.
    2. plan        : greedy-by-size offsets for the buffers of that log (tensor/plan.py).
    3. packed pass : fn() runs again into the real graph. The n-th allocation is carved at
                     offset[n] of one arena instead of being first-fit allocated.

``fn`` therefore runs twice and must allocate the same tensors in the same order both times.
That is checked: the allocation cursor of the packed pass has to end at the number of
planned buffers, otherwise RuntimeError.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

from . import plan as planmod

logger = logging.getLogger("cuda_graph_memopt")

_SYMBOLS = (
    "tmo_set_capture_plan",
    "tmo_set_capture_plan_at",
    "tmo_capture_plan_arena_base",
    "tmo_clear_capture_plan",
    "tmo_capture_plan_cursor",
    "tmo_trace_begin",
    "tmo_trace_end",
    "tmo_trace_count",
    "tmo_copy_trace",
    "tmo_va_only_begin",
    "tmo_va_only_end",
    "tmo_vmm_commit",
    "tmo_vmm_physical_bytes",
)


def loaded_c10_cuda_path() -> Optional[str]:
    """Path of the libc10_cuda.so mapped into this process (from /proc/self/maps)."""
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                if "libc10_cuda.so" in line:
                    return line.split()[-1]
    except OSError:
        pass
    return None


def load_lib():
    """Open the patched libc10_cuda.so that torch already loaded and declare its prototypes."""
    lib = ctypes.CDLL("libc10_cuda.so")
    missing = [s for s in _SYMBOLS if not hasattr(lib, s)]
    if missing:
        raise RuntimeError(
            "libc10_cuda.so lacks %s. It is the stock library loaded at %s. Build the patched one with "
            "cuda_graph_memopt/build.sh c10 and put its directory first in LD_LIBRARY_PATH."
            % (missing[:3], loaded_c10_cuda_path())
        )
    sz = ctypes.POINTER(ctypes.c_size_t)
    lib.tmo_set_capture_plan.argtypes = [sz, sz, ctypes.c_size_t, ctypes.c_size_t]
    lib.tmo_set_capture_plan.restype = None
    lib.tmo_set_capture_plan_at.argtypes = [sz, sz, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
    lib.tmo_set_capture_plan_at.restype = None
    lib.tmo_capture_plan_arena_base.restype = ctypes.c_void_p
    lib.tmo_capture_plan_cursor.restype = ctypes.c_size_t
    lib.tmo_trace_count.restype = ctypes.c_size_t
    lib.tmo_copy_trace.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.tmo_copy_trace.restype = None
    for name in ("tmo_clear_capture_plan", "tmo_trace_begin", "tmo_trace_end",
                 "tmo_va_only_begin", "tmo_va_only_end", "tmo_vmm_commit"):
        getattr(lib, name).restype = None
    lib.tmo_vmm_physical_bytes.restype = ctypes.c_size_t
    return lib


def _read_trace(lib) -> list[dict]:
    """Events the private-pool filter accepted (see CUDACachingAllocator.cpp note_tmo_trace)."""
    n = int(lib.tmo_trace_count())
    if n == 0:
        return []
    actions = (ctypes.c_uint8 * n)()
    addrs = (ctypes.c_uint64 * n)()
    sizes = (ctypes.c_uint64 * n)()
    lib.tmo_copy_trace(actions, addrs, sizes, n)
    return [
        {
            "action": "alloc" if actions[i] == 1 else "free_completed",
            "addr": int(addrs[i]),
            "size": int(sizes[i]),
        }
        for i in range(n)
    ]


def _pool_key(pool) -> Optional[tuple]:
    if pool is None:
        return None
    if isinstance(pool, (tuple, list)):
        return tuple(pool)
    return ("id", id(pool))


def _segments_of(events: list[dict]) -> list[dict]:
    """Caching-allocator segments (address, size, stream) that contain a traced address."""
    addrs = sorted({e["addr"] for e in events})
    segs = []
    for seg in torch.cuda.memory._snapshot().get("segments", []):
        base, size = int(seg["address"]), int(seg["total_size"])
        i = _bisect_left(addrs, base)
        if i < len(addrs) and addrs[i] < base + size:
            segs.append({"address": base, "size": size, "stream": int(seg.get("stream", 0)),
                         "n_blocks": len(seg.get("blocks", []))})
    segs.sort(key=lambda s: s["address"])
    return segs


def _bisect_left(a, x):
    import bisect

    return bisect.bisect_left(a, x)


@dataclass
class TensorResult:
    mode: str  # packed | reuse_pool | no_allocations | fallback
    report: dict = field(default_factory=dict)
    # Debug artifacts, only filled when requested.
    events: list = field(default_factory=list)
    buffers: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    arena_bytes: int = 0


def _new_record_graph() -> "torch.cuda.CUDAGraph":
    # keep_graph=True: the throw-away graph is never instantiated, so it costs no exec memory.
    try:
        return torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        return torch.cuda.CUDAGraph()


@dataclass
class _Arena:
    base: int
    nbytes: int


# One arena per private pool. Process-wide: several runners (target, draft) share the global graph
# pool and must not plant a second arena in it. SGLang records the largest batch first; every later
# capture on the pool is planned *into* that arena (its free spans), so that the pool never grows
# the way online first-fit would make it grow.
_ARENAS: dict[tuple, _Arena] = {}


def _arena_occupied(base: int) -> Optional[tuple[list[tuple[int, int]], int]]:
    """(offset, size) of every block of the arena segment that is not free, and the segment size."""
    for seg in torch.cuda.memory._snapshot().get("segments", []):
        if int(seg["address"]) != base:
            continue
        offset, busy = 0, []
        for blk in seg.get("blocks", []):
            size = int(blk["size"])
            if blk.get("state") != "inactive":
                busy.append((offset, size))
            offset += size
        return busy, int(seg["total_size"])
    return None


class TensorPlanner:
    """Library handle plus the planning logic (the arenas are process-wide, see ``_ARENAS``)."""

    def __init__(self, use_vmm: bool = True, strict: bool = False, slack: float = 0.05):
        self.lib = load_lib()
        self.use_vmm = use_vmm
        self.strict = strict
        # The first capture of a pool is packed unless the arena is more than (1 + slack) x what
        # first-fit took for it. Packing pays off across the captures that follow on the same pool,
        # which a single record pass cannot see.
        self.slack = float(slack)
        self.arenas = _ARENAS
        logger.info("tensor: libc10_cuda=%s vmm=%s", loaded_c10_cuda_path(), use_vmm)

    # ------------------------------------------------------------------
    def _plain(self, ctx, graph, pool, stream, fn):
        with ctx(cuda_graph=graph, pool=pool, stream=stream):
            return fn()

    def capture(
        self,
        graph,
        pool,
        stream,
        fn: Callable[[], Any],
        graph_ctx: Optional[Callable] = None,
        artifacts: bool = False,
    ) -> tuple[Any, TensorResult]:
        """Capture ``fn`` into ``graph``; returns (fn's return value, TensorResult)."""
        key = _pool_key(pool)
        ctx = graph_ctx or torch.cuda.graph
        stream_value = int(stream.cuda_stream)
        lib = self.lib

        arena = self.arenas.get(key) if key is not None else None
        occupied: list[tuple[int, int]] = []
        if arena is not None:
            found = _arena_occupied(arena.base)
            if found is None:  # the segment was released (pool destroyed and recreated)
                del self.arenas[key]
                arena = None
            else:
                occupied = found[0]

        # ---- 1. record pass -------------------------------------------------------------
        t0 = time.perf_counter()
        lib.tmo_trace_begin()
        record = _new_record_graph()
        try:
            with torch.cuda.graph(record, stream=stream):
                # torch.cuda.graph.__enter__ runs empty_cache(): take the baseline after it, so
                # the delta is what first-fit grew the private pool by while running fn().
                reserved0 = torch.cuda.memory_reserved()
                recorded = fn()
                record_reserved = torch.cuda.memory_reserved() - reserved0
            events = _read_trace(lib)
        finally:
            lib.tmo_trace_end()
        segments = _segments_of(events) if artifacts else []
        record_segment_bytes = sum(s["size"] for s in segments) if artifacts else None

        # ---- 2. plan --------------------------------------------------------------------
        fixed = [planmod.fixed_buffer(i, off, size) for i, (off, size) in enumerate(occupied)]
        planned = planmod.plan_from_events(events, fixed=fixed, ceiling=arena.nbytes if arena else None)
        arena_bytes = arena.nbytes if arena is not None else planned["arena_bytes"]
        report: dict[str, Any] = {
            "stream": stream_value,
            "n_events": len(events),
            "n_buffers": planned["n_buffers"],
            "peak_live_bytes": planned["peak_live_bytes"],
            "packed_bytes": planned["packed_bytes"],
            "arena_bytes": arena_bytes,
            "record_reserved_delta": record_reserved,
            "record_segment_bytes": record_segment_bytes,
            "into_existing_arena": arena is not None,
            "arena_obstacles": len(occupied),
            "record_ms": (time.perf_counter() - t0) * 1e3,
        }
        if artifacts:
            planmod.verify_plan(planned["buffers"] + fixed, arena_bytes)
        del recorded
        gc.collect()
        record.reset()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        res = TensorResult("packed", report, arena_bytes=arena_bytes)
        if artifacts:
            res.events = events
            res.buffers = planned["buffers"]
            res.segments = segments

        if planned["n_buffers"] == 0:
            res.mode = "no_allocations"
            return self._plain(ctx, graph, pool, stream, fn), res
        if arena is not None and planned["packed_bytes"] > arena.nbytes:
            # Larger than the arena (or its free spans): first-fit takes what it needs from the pool.
            res.mode = "reuse_pool"
            report["reason"] = "does not fit the arena"
            return self._plain(ctx, graph, pool, stream, fn), res
        if arena is None and planned["arena_bytes"] > record_reserved * (1.0 + self.slack):
            # The arena (one whole segment) is clearly larger than what the caching allocator took
            # in the record pass: keep the stock behaviour.
            res.mode = "skipped_no_gain"
            return self._plain(ctx, graph, pool, stream, fn), res

        # ---- 3. packed pass -------------------------------------------------------------
        t1 = time.perf_counter()
        n = planned["n_buffers"]
        sizes = (ctypes.c_size_t * n)(*planned["rounded_sizes"])
        offsets = (ctypes.c_size_t * n)(*planned["offsets"])
        new_arena = arena is None
        vmm = self.use_vmm and new_arena
        if new_arena:
            lib.tmo_set_capture_plan(sizes, offsets, n, arena_bytes)
        else:
            lib.tmo_set_capture_plan_at(sizes, offsets, n, arena_bytes, arena.base)
        reserved1 = torch.cuda.memory_reserved()
        if vmm:
            lib.tmo_va_only_begin()
        try:
            with ctx(cuda_graph=graph, pool=pool, stream=stream):
                out = fn()
            cursor = int(lib.tmo_capture_plan_cursor())
        finally:
            if vmm:
                lib.tmo_va_only_end()
            lib.tmo_clear_capture_plan()
            if vmm:
                lib.tmo_vmm_commit()
        torch.cuda.synchronize()
        report.update(
            cursor=cursor,
            packed_reserved_delta=torch.cuda.memory_reserved() - reserved1,
            physical_bytes=int(lib.tmo_vmm_physical_bytes()),
            packed_ms=(time.perf_counter() - t1) * 1e3,
            vmm=vmm,
        )
        if cursor != n:
            raise RuntimeError(
                f"cuda_graph_memopt: the packed capture issued {cursor} allocations but the record pass "
                f"planned {n}. fn() is not deterministic across calls."
            )
        if new_arena and key is not None:
            base = lib.tmo_capture_plan_arena_base()
            if base:
                self.arenas[key] = _Arena(int(base), arena_bytes)
        return out, res
