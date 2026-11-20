"""GraphMemOpt: the public entry point of cuda_graph_memopt.

    opt = GraphMemOpt.from_env()            # None when CGMO is unset: nothing changes
    graph = opt.new_graph()
    out = opt.capture(graph, pool, stream, fn, key="0_128", group=0)   # instead of `with torch.cuda.graph`
    ...
    opt.verify_pending()                    # after every graph_capture() context has exited
    opt.replay("0_128", graph)              # instead of graph.replay()

``capture`` does, in order (each step only when its feature is enabled):

    tensor : record pass -> plan -> packed capture into ``graph``      (tensor/runtime.py)
    fold   : turn the periodic layer region of ``graph`` into a WHILE node (fold/graph_fold.cu)
             otherwise: instantiate the unrolled graph as usual

Any step that cannot be applied falls back to the stock behaviour for that graph (warning in the
log), unless ``CGMO_STRICT=1``. The output tensors, their addresses and the kernels executed are
what a plain ``torch.cuda.graph`` capture would give, so callers keep using ``out`` unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Optional

import torch

from .config import Config

logger = logging.getLogger("cuda_graph_memopt")


def _driver_used() -> int:
    free, total = torch.cuda.mem_get_info()
    return total - free


def _tensors_of(obj) -> list:
    """Flatten the tensors of a graph output (tensor / dataclass / dict / .tensors)."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [t for f in dataclasses.fields(obj) for t in _tensors_of(getattr(obj, f.name))]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _tensors_of(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _tensors_of(v)]
    tensors = getattr(obj, "tensors", None)
    if isinstance(tensors, dict):
        return _tensors_of(tensors)
    return []


def _rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def _safe(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _pool_stats(pool) -> dict:
    """Segments of the private pool right now: how much memory the pool holds and how it is used."""
    try:
        pid = tuple(pool) if isinstance(pool, (tuple, list)) else None
        segs = [sg for sg in torch.cuda.memory._snapshot().get("segments", [])
                if pid is not None and tuple(sg.get("segment_pool_id", ())) == pid]
        return {
            "n_segments": len(segs),
            "reserved": sum(int(sg["total_size"]) for sg in segs),
            "allocated": sum(int(sg["allocated_size"]) for sg in segs),
            "segments": [{"size": int(sg["total_size"]), "allocated": int(sg["allocated_size"]),
                          "stream": int(sg.get("stream", 0)), "n_blocks": len(sg.get("blocks", []))}
                         for sg in sorted(segs, key=lambda x: -int(x["total_size"]))[:8]],
        }
    except Exception as exc:  # statistics must never break a capture
        return {"error": f"{type(exc).__name__}: {exc}"}


class GraphMemOpt:
    _n_instances = 0  # one manager per CUDA graph runner; dump directories must not collide

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.instance = GraphMemOpt._n_instances
        GraphMemOpt._n_instances += 1
        self.planner = None
        self.fold_lib = None
        self.folded: dict = {}
        self.refused: dict = {}
        self.records: list[dict] = []
        self.verify_results: list[dict] = []
        self._verified: dict = {}
        self._pending: list = []
        self._seq = 0
        self._dumped = 0
        self._dumped_groups: set = set()
        self._rank = _rank()
        self.proc_dir: Optional[str] = None
        self._setup_logging()
        logger.info("cuda_graph_memopt config: %s", cfg.as_dict())
        self._init_features()

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls) -> Optional["GraphMemOpt"]:
        cfg = Config.from_env()
        return cls(cfg) if cfg.enabled else None

    def _setup_logging(self) -> None:
        if self.cfg.debug >= 1:
            self.proc_dir = os.path.join(self.cfg.dump_dir, f"rank{self._rank}_pid{os.getpid()}")
            os.makedirs(self.proc_dir, exist_ok=True)
        if getattr(logger, "_cgmo_ready", False):
            return
        logger.setLevel(logging.DEBUG if self.cfg.debug >= 1 else logging.INFO)
        fmt = logging.Formatter(f"[cgmo r{self._rank}] %(levelname)s %(message)s")
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if self.proc_dir:
            fh = logging.FileHandler(os.path.join(self.proc_dir, "cgmo.log"))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
        logger.propagate = False
        logger._cgmo_ready = True  # type: ignore[attr-defined]

    def _init_features(self) -> None:
        cfg = self.cfg
        features = list(cfg.features)
        if cfg.tensor:
            try:
                from .tensor.runtime import TensorPlanner

                self.planner = TensorPlanner(use_vmm=cfg.tensor_vmm, strict=cfg.strict, slack=cfg.tensor_slack)
            except Exception as exc:
                if cfg.strict:
                    raise
                logger.error("tensor feature disabled: %s", exc)
                features.remove("tensor")
        if cfg.fold:
            try:
                from .fold.lib import GraphFoldLib

                self.fold_lib = GraphFoldLib.get(cfg.lib_dir)
                torch.cuda.CUDAGraph(keep_graph=True)  # raises TypeError on torch without keep_graph
            except Exception as exc:
                if cfg.strict:
                    raise
                logger.error("fold feature disabled: %s", exc)
                self.fold_lib = None
                features.remove("fold")
        self.active = tuple(features)

    # ------------------------------------------------------------------ graphs
    def new_graph(self) -> "torch.cuda.CUDAGraph":
        """CUDAGraph the capture can be folded from (keep_graph keeps the cudaGraph_t)."""
        if self.fold_lib is not None:
            return torch.cuda.CUDAGraph(keep_graph=True)
        return torch.cuda.CUDAGraph()

    # ------------------------------------------------------------------ dumping
    def _dump_dir_for(self, key, group, seq) -> Optional[str]:
        cfg = self.cfg
        if cfg.debug < 2 or self._dumped >= cfg.dump_max:
            return None
        if cfg.dump_ranks != "all" and str(self._rank) not in cfg.dump_ranks.split(","):
            return None
        if cfg.dump_filter:
            if not re.search(cfg.dump_filter, str(key)):
                return None
        elif group in self._dumped_groups:
            return None
        self._dumped_groups.add(group)
        self._dumped += 1
        d = os.path.join(self.proc_dir, f"capture_m{self.instance}_{seq:04d}_{_safe(key)}")
        os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------ capture
    def capture(
        self,
        graph,
        pool,
        stream,
        fn: Callable[[], Any],
        *,
        key,
        group=None,
        graph_ctx: Optional[Callable] = None,
    ):
        """Capture ``fn`` into ``graph`` (made by ``new_graph``) and prepare it for replay."""
        cfg = self.cfg
        seq = self._seq
        self._seq += 1
        dump = self._dump_dir_for(key, group, seq)
        rec: dict[str, Any] = {"seq": seq, "key": str(key), "group": group, "features": list(self.active),
                               "stream": int(stream.cuda_stream), "pool": list(pool) if isinstance(pool, (tuple, list)) else str(pool)}
        t0 = time.perf_counter()
        used0, reserved0 = _driver_used(), torch.cuda.memory_reserved()
        ctx = graph_ctx or torch.cuda.graph

        # ---- tensor placement or plain capture
        tensor_info = None
        out = None
        if self.planner is not None:
            try:
                out, tres = self.planner.capture(graph, pool, stream, fn, graph_ctx,
                                                 artifacts=dump is not None)
                rec["tensor"] = dict(tres.report, mode=tres.mode)
                if dump is not None:
                    from .debug import dump as dbg

                    tensor_info = dbg.dump_tensor(tres, os.path.join(dump, "tensor"))
                elif cfg.debug >= 1:
                    tensor_info = dict(tres.report, mode=tres.mode)
            except Exception as exc:
                if cfg.strict:
                    raise
                logger.error("tensor placement failed for %s (%s: %s); plain capture", key, type(exc).__name__, exc)
                rec["tensor"] = {"mode": "fallback", "error": f"{type(exc).__name__}: {exc}"}
                graph.reset()
                torch.cuda.synchronize()
                out = None
        if out is None and (self.planner is None or rec.get("tensor", {}).get("mode") == "fallback"):
            with ctx(cuda_graph=graph, pool=pool, stream=stream):
                out = fn()
        rec["capture_ms"] = (time.perf_counter() - t0) * 1e3

        # ---- fold or instantiate
        fold_info = None
        if self.fold_lib is not None:
            fdir = os.path.join(dump, "fold") if dump else None
            if fdir:
                os.makedirs(fdir, exist_ok=True)
            used1 = _driver_used()
            t1 = time.perf_counter()
            folded, info = self.fold_lib.fold(graph.raw_cuda_graph(), stream.cuda_stream,
                                              cfg.fold_min_iters, cfg.fold_min_saved_nodes, fdir)
            info["fold_wall_ms"] = (time.perf_counter() - t1) * 1e3
            info["driver_used_delta_fold"] = _driver_used() - used1
            if folded is None:
                if cfg.strict:
                    raise RuntimeError(f"fold refused for {key}: {info['message']}")
                self.refused[str(key)] = info["message"]
                logger.warning("fold refused for %s: %s", key, info["message"])
                graph.instantiate()
                rec["fold"] = {"status": "refused", "message": info["message"]}
            else:
                self.folded[key] = folded
                rec["fold"] = {**info, "status": "folded"}
                if self._verified.get(group, 0) < cfg.verify:
                    # Replay inside the capture context is illegal (custom all-reduce registers
                    # the graph's IPC buffers when it exits), so the check is queued.
                    self._verified[group] = self._verified.get(group, 0) + 1
                    self._pending.append((key, graph, folded, out, stream))
            if fdir:
                from .debug import dump as dbg

                fold_info = dbg.dump_fold_post(fdir, info, rec["fold"]["status"])

        if cfg.debug >= 1:
            rec["pool_after"] = _pool_stats(pool)
        rec["driver_delta"] = _driver_used() - used0
        rec["reserved_delta"] = torch.cuda.memory_reserved() - reserved0
        rec["wall_ms"] = (time.perf_counter() - t0) * 1e3
        self.records.append(rec)
        (logger.info if cfg.debug >= 1 else logger.debug)(
            "capture %s group=%s: driver +%.1f MiB, reserved +%.1f MiB, %s%s%.0f ms", key, group,
            rec["driver_delta"] / 2**20, rec["reserved_delta"] / 2**20,
            f"tensor={rec['tensor']['mode']} " if "tensor" in rec else "",
            f"fold={rec['fold']['status']} " if "fold" in rec else "", rec["wall_ms"])
        if self.proc_dir:
            with open(os.path.join(self.proc_dir, "captures.jsonl"), "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        if dump:
            from .debug import dump as dbg

            meta = dict(rec, config=cfg.as_dict(), dump_dir=dump)
            dbg.write_report(dump, meta, tensor_info, fold_info)
            logger.info("dumped capture %s to %s", key, dump)
        return out

    # ------------------------------------------------------------------ replay
    def replay(self, key, graph) -> None:
        folded = self.folded.get(key)
        if folded is None:
            graph.replay()
        else:
            folded.launch(torch.cuda.current_stream().cuda_stream)

    def has_folded(self, key) -> bool:
        return key in self.folded

    # ------------------------------------------------------------------ verification
    def verify_pending(self) -> None:
        """Run queued fold checks. Call after every graph_capture() context has exited."""
        pending, self._pending = self._pending, []
        for key, graph, folded, output, stream in pending:
            try:
                self._verify(key, graph, folded, output, stream)
            except Exception as exc:
                if self.cfg.strict:
                    raise
                logger.error("verify of %s failed: %s: %s", key, type(exc).__name__, exc)

    def _verify(self, key, graph, folded, output, stream) -> None:
        """Replay the unrolled and the folded executable on the same inputs; compare bitwise."""

        def sync():
            # torch.cuda.synchronize() does not wait for work on a green-context stream
            stream.synchronize()
            torch.cuda.synchronize()

        def run_orig():
            with torch.cuda.stream(stream):
                graph.replay()
            sync()

        def run_fold():
            folded.launch(stream.cuda_stream)
            sync()

        tensors = _tensors_of(output)
        graph.instantiate()  # the unrolled executable only exists for this check
        sync()
        run_orig()
        ref1 = [t.clone() for t in tensors]
        sync()
        run_orig()
        ref2 = [t.clone() for t in tensors]
        sync()
        with torch.cuda.stream(stream):
            for t in tensors:
                t.zero_()
        sync()
        run_fold()
        got = [t.clone() for t in tensors]
        sync()

        def same(a, b):
            return all(
                x.shape == y.shape
                and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
                for x, y in zip(a, b)
            )

        def maxdiff(a, b):
            m = 0.0
            for x, y in zip(a, b):
                if x.is_floating_point():
                    d = (x.float() - y.float()).abs()
                    m = max(m, float(torch.nan_to_num(d, nan=-1.0).max()))
            return m

        def timeit(fn, reps=20):
            for _ in range(3):
                fn()
            sync()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                e0.record(stream)
                for _ in range(reps):
                    fn()
                e1.record(stream)
            stream.synchronize()
            return e0.elapsed_time(e1) / reps

        def orig_async():
            with torch.cuda.stream(stream):
                graph.replay()

        orig_ms = timeit(orig_async)
        fold_ms = timeit(lambda: folded.launch(stream.cuda_stream))
        orig_ms2 = timeit(orig_async)
        res = {
            "key": str(key), "n_tensors": len(tensors),
            "orig_ms": orig_ms, "orig_ms_repeat": orig_ms2, "fold_ms": fold_ms,
            "fold_over_orig": fold_ms / (0.5 * (orig_ms + orig_ms2)),
            "orig_vs_orig_bitwise": same(ref1, ref2),
            "fold_vs_orig_bitwise": same(ref1, got),
            "orig_vs_orig_maxdiff": maxdiff(ref1, ref2),
            "fold_vs_orig_maxdiff": maxdiff(ref1, got),
            "state": folded.state(),
        }
        self.verify_results.append(res)
        logger.info("fold verify %s", res)
        if self.cfg.strict and not res["fold_vs_orig_bitwise"] and res["orig_vs_orig_bitwise"]:
            raise RuntimeError(f"folded graph {key} is not bitwise equal to the unrolled graph: {res}")

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        folded = [r["fold"] for r in self.records if r.get("fold", {}).get("status") == "folded"]
        tens = [r["tensor"] for r in self.records if "tensor" in r]
        tot = lambda rows, k: sum((r.get(k) or 0) for r in rows)
        return {
            "config": self.cfg.as_dict(),
            "active_features": list(self.active),
            "captures": len(self.records),
            "driver_delta_total": sum(r["driver_delta"] for r in self.records),
            "reserved_delta_total": sum(r["reserved_delta"] for r in self.records),
            "capture_wall_ms_total": sum(r["wall_ms"] for r in self.records),
            "tensor": {
                "modes": {m: sum(1 for t in tens if t.get("mode") == m) for m in {t.get("mode") for t in tens}},
                "arena_bytes_total": tot(tens, "arena_bytes"),
                "record_segment_bytes_total": tot(tens, "record_segment_bytes"),
                "peak_live_bytes_total": tot(tens, "peak_live_bytes"),
            } if tens else None,
            "fold": {
                "folded_graphs": len(folded),
                "refused_graphs": len(self.refused),
                "refused_reasons": sorted(set(self.refused.values())),
                "unrolled_nodes_total": tot(folded, "unrolled_nodes"),
                "folded_parent_nodes_total": tot(folded, "folded_nodes_actual"),
                "table_bytes_total": tot(folded, "table_bytes"),
                "instantiate_driver_delta_total": tot(folded, "instantiate_driver_delta"),
                "example": folded[0] if folded else None,
            } if self.fold_lib is not None else None,
            "verify": self.verify_results,
        }

    def finish(self) -> dict:
        """Log and (debug >= 1) write the summary. Call once after the last capture."""
        s = self.summary()
        logger.info("summary: %s", json.dumps({k: v for k, v in s.items() if k not in ("config", "verify")}, default=str))
        if self.proc_dir:
            with open(os.path.join(self.proc_dir, "summary.json"), "w") as fh:
                json.dump(s, fh, indent=1, default=str)
        return s

    def close(self) -> None:
        for f in self.folded.values():
            f.close()
        self.folded.clear()
