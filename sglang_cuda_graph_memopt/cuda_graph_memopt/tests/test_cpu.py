#!/usr/bin/env python3
"""CPU-only tests of cuda_graph_memopt: planner, config parsing, dump post-processing.

    python3 test_cpu.py            # prints one line per check, exit code 1 on any failure
Writes results/cpu_tests.json.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import time
import traceback
import xml.dom.minidom

ROOT = os.environ.get("CGMO_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
sys.path.insert(0, ROOT)

from cuda_graph_memopt.config import Config, parse_features  # noqa: E402
from cuda_graph_memopt.debug import dump as dbg  # noqa: E402
from cuda_graph_memopt.tensor import plan as P  # noqa: E402

RESULTS = []


def check(name):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            RESULTS.append({"name": name, "pass": True, "detail": detail, "ms": (time.perf_counter() - t0) * 1e3})
            print(f"PASS {name}" + (f"  [{detail}]" if detail else ""))
        except Exception:
            RESULTS.append({"name": name, "pass": False, "detail": traceback.format_exc(), "ms": (time.perf_counter() - t0) * 1e3})
            print(f"FAIL {name}\n{traceback.format_exc()}")
        return fn

    return deco


def trace(spec):
    """spec: list of ('a', addr, size) / ('f', addr)"""
    ev = []
    for s in spec:
        if s[0] == "a":
            ev.append({"action": "alloc", "addr": s[1], "size": s[2]})
        else:
            ev.append({"action": "free_completed", "addr": s[1], "size": 0})
    return ev


def random_trace(rng, n_ops, max_mib=64, p_free=0.5):
    ev, live, addr = [], [], 0x1000
    for _ in range(n_ops):
        if live and rng.random() < p_free:
            ev.append({"action": "free_completed", "addr": live.pop(rng.randrange(len(live))), "size": 0})
        else:
            size = rng.choice([rng.randint(1, 4096), rng.randint(1, 1 << 20), rng.randint(1, max_mib << 20)])
            addr += 1 << 22
            ev.append({"action": "alloc", "addr": addr, "size": size})
            live.append(addr)
    return ev


@check("round_size / segment_size follow the caching allocator")
def _():
    assert P.round_size(1) == 512 and P.round_size(512) == 512 and P.round_size(513) == 1024
    assert P.segment_size(1) == 2 << 20 and P.segment_size(1 << 20) == 2 << 20
    assert P.segment_size((1 << 20) + 1) == 20 << 20 and P.segment_size(10 << 20) == 10 << 20
    assert P.segment_size((10 << 20) + 1) == 12 << 20
    try:
        P.round_size(0)
    except ValueError:
        return "0-byte request rejected"
    raise AssertionError("0-byte request accepted")


@check("lifetimes: free_completed ends a buffer, unknown free ignored, live address reuse rejected")
def _():
    bufs = P.buffers_from_trace(trace([("a", 1, 1000), ("a", 2, 2000), ("f", 1), ("f", 99), ("a", 1, 3000)]))
    assert [b.end for b in bufs] == [2, None, None], [b.end for b in bufs]
    assert [b.start for b in bufs] == [0, 1, 3]
    try:
        P.buffers_from_trace(trace([("a", 1, 10), ("a", 1, 10)]))
    except RuntimeError:
        return "ok"
    raise AssertionError("double alloc of a live address accepted")


@check("free_requested never ends a lifetime")
def _():
    ev = trace([("a", 1, 4096)]) + [{"action": "free_requested", "addr": 1, "size": 4096}]
    b = P.buffers_from_trace(ev)
    assert b[0].end is None


@check("fragmentation pattern (big/small/big): packed span equals peak live")
def _():
    # big / small / big again: a first-fit allocator splits the freed big block for the small one
    MiB = 1 << 20
    ev = trace([("a", 1, 32 * MiB), ("a", 2, 4 * MiB), ("f", 1), ("a", 3, 2 * MiB), ("a", 4, 32 * MiB), ("f", 2), ("f", 3)])
    plan = P.plan_from_events(ev)
    P.verify_plan(plan["buffers"], plan["arena_bytes"])
    assert plan["packed_bytes"] <= 38 * MiB, plan["packed_bytes"]
    return f"peak {plan['peak_live_bytes'] / MiB:.0f} MiB, packed {plan['packed_bytes'] / MiB:.0f} MiB"


@check("persistent buffers (end=None) never share addresses with later buffers")
def _():
    ev = trace([("a", 1, 4096), ("a", 2, 4096), ("f", 2), ("a", 3, 4096), ("a", 4, 8192)])
    plan = P.plan_from_events(ev)
    b = plan["buffers"]
    P.verify_plan(b, plan["arena_bytes"])
    for i in (0, 3):  # alive at the end
        for j in range(len(b)):
            if i != j and b[j].start > b[i].start:
                assert not (b[i].offset < b[j].offset + b[j].size and b[j].offset < b[i].offset + b[i].size)


@check("fuzz: 400 random traces, no-overlap + bounds + alignment + peak <= packed <= sum")
def _():
    rng = random.Random(7)
    worst = 0.0
    n_total = 0
    for seed in range(400):
        ev = random_trace(rng, rng.randint(1, 250), p_free=rng.choice([0.2, 0.5, 0.8]))
        plan = P.plan_from_events(ev)
        bufs = plan["buffers"]
        n_total += len(bufs)
        if not bufs:
            assert plan["packed_bytes"] == 0 and plan["arena_bytes"] == 0
            continue
        v = P.verify_plan(bufs, plan["arena_bytes"])
        assert plan["peak_live_bytes"] <= plan["packed_bytes"] <= sum(b.size for b in bufs)
        assert plan["arena_bytes"] >= plan["packed_bytes"]
        assert P.segment_size(plan["arena_bytes"]) == plan["arena_bytes"] and plan["arena_bytes"] == P.arena_size(plan["packed_bytes"])
        worst = max(worst, plan["packed_bytes"] / plan["peak_live_bytes"])
    return f"{n_total} buffers; worst packed/peak = {worst:.3f}"


@check("fuzz: planning around fixed obstacles (arena already used by earlier captures)")
def _():
    rng = random.Random(23)
    n_cases = 0
    for seed in range(200):
        ev = random_trace(rng, rng.randint(1, 120), max_mib=16, p_free=rng.choice([0.3, 0.6]))
        # obstacles: disjoint spans at random 512-aligned offsets
        fixed, cursor = [], 0
        for i in range(rng.randint(0, 6)):
            cursor += 512 * rng.randint(0, 40000)
            size = 512 * rng.randint(1, 20000)
            fixed.append(P.fixed_buffer(i, cursor, size))
            cursor += size
        plan = P.plan_from_events(ev, fixed=fixed)
        bufs = plan["buffers"]
        if not bufs:
            continue
        n_cases += 1
        span = max(max(b.offset + b.size for b in bufs), max((f.offset + f.size for f in fixed), default=0))
        P.verify_plan(bufs + fixed, span)  # no new buffer overlaps an obstacle or a live neighbour
        for f in fixed:
            for b in bufs:
                assert b.offset + b.size <= f.offset or f.offset + f.size <= b.offset
        # without obstacles the same trace packs at least as tightly
        free = P.plan_from_events(ev)
        assert free["packed_bytes"] <= plan["packed_bytes"] or not fixed
    return f"{n_cases} traces with obstacles"


@check("captures of varying size planned one after another into one arena: small outputs never block later plans")
def _():
    MiB = 1 << 20
    rng = random.Random(5)
    def trace_k(k):  # growing-size pattern, two live at a time, plus one 4 KiB output that survives
        ev, n = [], 0
        prev = None
        for i in range(k):
            ev.append({"action": "alloc", "addr": 0x1000 * (n + 1), "size": (12 + i) * MiB}); n += 1
            if prev is not None:
                ev.append({"action": "free_completed", "addr": prev, "size": 0})
            prev = 0x1000 * n
        ev.append({"action": "free_completed", "addr": prev, "size": 0})
        ev.append({"action": "alloc", "addr": 0x1000 * (n + 1), "size": 4096})
        return ev
    first = P.plan_from_events(trace_k(40))
    arena = first["arena_bytes"]
    busy = [P.fixed_buffer(0, b.offset, b.size) for b in first["buffers"] if b.end is None]
    for step, k in enumerate([40, 32, 40, 20, 40, 40]):
        plan = P.plan_from_events(trace_k(k), fixed=busy, ceiling=arena)
        assert plan["packed_bytes"] <= arena, f"capture {step} (k={k}) does not fit: {plan['packed_bytes']} > {arena}"
        P.verify_plan(plan["buffers"] + busy, arena)
        busy += [P.fixed_buffer(len(busy) + i, b.offset, b.size) for i, b in enumerate(plan["buffers"]) if b.end is None]
    return f"arena {arena / MiB:.1f} MiB, {len(busy)} surviving outputs"


@check("fuzz at SGLang scale: 2200 buffers, 60 s budget for plan + verify")
def _():
    rng = random.Random(11)
    ev = random_trace(rng, 2600, max_mib=40, p_free=0.45)
    t0 = time.perf_counter()
    plan = P.plan_from_events(ev)
    t1 = time.perf_counter()
    P.verify_plan(plan["buffers"], plan["arena_bytes"])
    t2 = time.perf_counter()
    assert t2 - t0 < 60
    return f"{plan['n_buffers']} buffers: plan {t1 - t0:.2f} s, verify {t2 - t1:.2f} s"


@check("arena_size is a fixed point of segment_size (tiny spans map to the 20 MiB large segment)")
def _():
    assert P.arena_size(100 << 10) == 20 << 20      # 2 MiB request would go to the large pool
    assert P.arena_size(3 << 20) == 20 << 20
    assert P.arena_size(10 << 20) == 10 << 20
    assert P.arena_size((10 << 20) + 1) == 12 << 20
    for n in list(range(1, 5000, 37)) + [1 << k for k in range(10, 36)]:
        a = P.arena_size(n)
        assert a >= n and P.segment_size(a) == a, n
    return "ok"


@check("verify_plan rejects overlap, misalignment, out-of-arena")
def _():
    plan = P.plan_from_events(trace([("a", 1, 4096), ("a", 2, 4096)]))
    b = plan["buffers"]
    for mutate, label in (
        (lambda: setattr(b[1], "offset", b[0].offset), "overlap"),
        (lambda: setattr(b[1], "offset", b[0].offset + 100), "misaligned"),
        (lambda: setattr(b[1], "offset", plan["arena_bytes"]), "outside"),
    ):
        old = b[1].offset
        mutate()
        try:
            P.verify_plan(b, plan["arena_bytes"])
        except AssertionError:
            b[1].offset = old
            continue
        raise AssertionError(f"verify_plan accepted a {label} plan")
    P.verify_plan(b, plan["arena_bytes"])
    return "3 mutations rejected"


@check("config: CGMO feature parsing and defaults")
def _():
    assert parse_features(None) == () and parse_features("none") == () and parse_features("0") == ()
    assert parse_features("all") == ("tensor", "fold")
    assert parse_features("fold,tensor") == ("tensor", "fold")
    assert parse_features("tensor+fold") == ("tensor", "fold")
    try:
        parse_features("tensr")
    except ValueError:
        pass
    else:
        raise AssertionError("typo accepted")
    for k in [k for k in os.environ if k.startswith("CGMO")]:
        del os.environ[k]
    c = Config.from_env()
    assert not c.enabled and c.tensor_vmm and c.fold_min_iters == 4 and not c.strict and c.debug == 0
    os.environ.update(CGMO="tensor", CGMO_TENSOR_VMM="0", CGMO_DEBUG="2", CGMO_STRICT="1")
    c = Config.from_env()
    assert c.tensor and not c.fold and not c.uses_vmm and c.strict and c.debug == 2
    from cuda_graph_memopt import vmm_arena_active

    assert vmm_arena_active() is False
    os.environ["CGMO_TENSOR_VMM"] = "1"
    assert vmm_arena_active() is True
    for k in [k for k in os.environ if k.startswith("CGMO")]:
        del os.environ[k]


class FakeRes:
    def __init__(self):
        rng = random.Random(5)
        ev = random_trace(rng, 160, max_mib=24)
        plan = P.plan_from_events(ev)
        self.mode = "packed"
        self.report = {k: plan[k] for k in ("n_buffers", "peak_live_bytes", "packed_bytes", "arena_bytes")}
        self.report.update(stream=1, n_events=len(ev), record_reserved_delta=3 << 30, record_segment_bytes=3 << 30,
                           physical_bytes=plan["arena_bytes"], packed_reserved_delta=plan["arena_bytes"])
        self.events, self.buffers, self.arena_bytes = ev, plan["buffers"], plan["arena_bytes"]
        addrs = sorted({e["addr"] for e in ev})
        self.segments = [{"address": a, "size": 1 << 22, "stream": 1, "n_blocks": 1} for a in addrs]


@check("debug dump: tensor artifacts (trace, csv, plan.json, svg, timeline) are complete and parse")
def _():
    res = FakeRes()
    with tempfile.TemporaryDirectory() as d:
        info = dbg.dump_tensor(res, os.path.join(d, "tensor"))
        for f in ("trace.jsonl", "buffers.csv", "plan.json", "layout.svg", "timeline.txt", "segments_before.json"):
            assert os.path.getsize(os.path.join(d, "tensor", f)) > 0, f
        xml.dom.minidom.parse(os.path.join(d, "tensor", "layout.svg"))
        n_rect = open(os.path.join(d, "tensor", "layout.svg")).read().count("<rect")
        assert n_rect >= 2 * len(res.buffers), n_rect
        rows = open(os.path.join(d, "tensor", "buffers.csv")).read().strip().split("\n")
        assert len(rows) == len(res.buffers) + 1
        assert info["verify"]["buffers"] == len(res.buffers)
        dbg.write_report(d, {"key": "k", "seq": 0, "features": ["tensor"], "wall_ms": 1.0}, info, None)
        md = open(os.path.join(d, "report.md")).read()
        assert "peak live" in md and "arena" in md
    return f"{len(res.buffers)} buffers, {n_rect} svg rects"


@check("debug dump: fold digest from the driver json (classes, region, updates, dot)")
def _():
    with tempfile.TemporaryDirectory() as d:
        def node(i, t="KERNEL", sym="_Z3fooIfEvPT_"):
            n = {"id": i, "type": t}
            if t == "KERNEL":
                n.update(symbol=sym, grid=[1, 1, 1], block=[128, 1, 1], smem=0, param_bytes=16, param_hash="00", param_head="00")
            return n

        before = {"label": "b", "n_nodes": 4, "nodes": [node(i) for i in range(4)], "n_edges": 3, "edges": [[0, 1, 0], [1, 2, 0], [2, 3, 0]]}
        after = {"label": "a", "n_nodes": 2, "nodes": [node(0), node(1, "CONDITIONAL")], "n_edges": 1, "edges": [[0, 1, 0]]}
        json.dump(before, open(os.path.join(d, "graph_before.json"), "w"))
        json.dump(after, open(os.path.join(d, "graph_after_parent.json"), "w"))
        json.dump(after, open(os.path.join(d, "graph_after_body.json"), "w"))
        json.dump({"n_kernels": 4, "n_classes": 1, "classes": [{"id": 0, "count": 4, "symbol": "_Z3fooIfEvPT_", "grid": [1, 1, 1], "block": [128, 1, 1], "smem": 0, "param_bytes": 16}],
                   "sequence_rle": [[0, 4]], "period_candidates": [{"P": 1, "start": 0, "iters": 4, "cover": 4}]}, open(os.path.join(d, "fold_classes.json"), "w"))
        json.dump({"region": {"start_kernel": 0, "period": 1, "iters": 4, "end_kernel_exclusive": 4, "first_gidx": 0, "last_gidx": 3, "prefix_kernels": 0, "suffix_kernels": 0, "loop_events_removed": 0},
                   "body_positions": [{"pos": 0, "class": 0, "gidx_iter0": 0, "symbol": "_Z3fooIfEvPT_", "grid": [1, 1, 1], "block": [128, 1, 1], "param_bytes": 16}],
                   "in_deps_gidx": [], "out_succ_gidx": [], "body_edges": [], "cross_iteration_edges": 3, "bypass_edges_added": 0,
                   "updates": [{"pos": 0, "offset": 0, "size": 8, "table_off": 0, "values": ["0100000000000000", "0200000000000000", "0300000000000000", "0400000000000000"]}],
                   "table": {"row_bytes": 16, "iters": 4, "total_bytes": 64}, "dry_run": False}, open(os.path.join(d, "fold_analysis.json"), "w"))
        digest = dbg.dump_fold_post(d, {"message": "folded", "n_nodes": 4, "n_kernels": 4, "n_edges": 3, "n_event_record": 0, "n_event_wait": 0, "n_classes": 1,
                                       "folded_nodes_actual": 2, "body_nodes": 2, "pdl_attr_dropped": 0, "instantiate_driver_delta": 0, "ctx_mode": 0}, "folded")
        for f in ("fold_summary.md", "graph_before.dot", "graph_after.dot", "graph_after_body.dot"):
            assert os.path.getsize(os.path.join(d, f)) > 0, f
        md = open(os.path.join(d, "fold_summary.md")).read()
        assert "period P = 1" in md and "launch classes" in md and "per-iteration parameter updates" in md
        assert "demangled" in open(os.path.join(d, "graph_before.json")).read()
        assert digest["status"] == "folded"


@check("demangle / short_name")
def _():
    s = dbg.demangle("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEEEEviT0_St5arrayIPcLm1EE")
    assert "vectorized_elementwise_kernel" in s, s
    short = dbg.short_name("_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEEEEviT0_St5arrayIPcLm1EE")
    assert short == "at::native::vectorized_elementwise_kernel<..>", short
    assert dbg.demangle("not_mangled") == "not_mangled"
    return short


def main():
    os.makedirs(os.environ.get("CGMO_TEST_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), exist_ok=True)
    with open(os.path.join(os.environ.get("CGMO_TEST_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), "cpu_tests.json"), "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    bad = [r["name"] for r in RESULTS if not r["pass"]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed")
    sys.exit(1 if bad else 0)


main()
