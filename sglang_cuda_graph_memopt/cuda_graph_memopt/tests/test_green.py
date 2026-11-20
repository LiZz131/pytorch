#!/usr/bin/env python3
"""Green Context test: stream A (36 SM) is optimised, stream B (96 SM) runs a peer workload.

    python3 test_green.py <mode>        mode: off | tensor | fold | both

Checks (one process per mode):
  1. the graph captured on A (inside A's green context) is replayed bitwise-equal to eager;
  2. partition kept: the kernels of the replay run on <= 36 distinct SMs (`mark` kernel records
     %smid), also when the replay is launched from B's stream;
  3. overlap: A's replay and B's graph run concurrently. hidden = (only_A + only_B - both) / only_A is
     the fraction of A's time hidden behind B; 0 = serialised, ~1 = fully overlapped. The same
     figure is derived from CUDA events on both streams (interval intersection).
Writes results/green/green_<mode>.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

MODE = sys.argv[1]
os.environ.pop("CGMO", None)
if MODE != "off":
    os.environ["CGMO"] = MODE if MODE != "both" else "tensor,fold"
os.environ.setdefault("CGMO_VERIFY", "0")

import torch  # noqa: E402

import cgmo_testlib as T  # noqa: E402
from cuda_graph_memopt import GraphMemOpt  # noqa: E402

OUT = os.path.join(T.OUT, "green")
os.makedirs(OUT, exist_ok=True)
LAYERS = int(os.environ.get("GF_LAYERS", "120"))  # GREEN_MODEL=chain|scratch (default scratch)


def main():
    res = {"mode": MODE, "layers": LAYERS, "pass": False}
    o = T.ops()
    A, A2, B, smA, smB = T.make_streams(True)
    res["sm_a_actual"], res["sm_b_actual"] = smA, smB
    seen = torch.zeros(256, dtype=torch.int32, device="cuda")
    model = (T.Model if os.environ.get("GREEN_MODEL") == "chain" else T.ScratchModel)(LAYERS, seen=seen)
    opt = GraphMemOpt.from_env() if MODE != "off" else None

    graph, out = T.capture_model(model, A, A2, opt, key="greenA", group=0)
    res.update(model.cap_stats)
    if opt:
        opt.verify_pending()
        res["fold_status"] = [r.get("fold", {}).get("status") for r in opt.records]
        res["tensor_mode"] = [r.get("tensor", {}).get("mode") for r in opt.records]

    def sync_all():
        T.sync_streams(A, A2, B)

    def seen_ids():
        sync_all()
        ids = [i for i, v in enumerate(seen.tolist()) if v]
        return {"count": len(ids), "min": min(ids) if ids else -1, "max": max(ids) if ids else -1}

    def run_a(stream=A):
        with torch.cuda.stream(stream):
            opt.replay("greenA", graph) if opt else graph.replay()

    # 1. numerics
    with torch.cuda.stream(A):
        model.x.copy_(torch.randn(T.N, device="cuda"))
    sync_all()
    with torch.cuda.stream(A):
        model.forward(A2)
    sync_all()
    ref = model.out.clone()
    model.out = out
    out.zero_()
    sync_all()
    run_a()
    sync_all()
    res["bitwise_vs_eager"] = bool(T.bitwise_equal(ref, out))

    # 2. partition kept
    seen.zero_()
    sync_all()
    run_a()
    res["sm_ids_replay_on_A"] = seen_ids()
    seen.zero_()
    sync_all()
    run_a(B)  # launched from the other partition's stream: the graph still belongs to A's context
    res["sm_ids_replay_launched_from_B"] = seen_ids()
    res["partition_kept"] = bool(res["sm_ids_replay_on_A"]["count"] <= smA and res["sm_ids_replay_launched_from_B"]["count"] <= smA)

    # 3. overlap with a 96-SM peer, burn length calibrated to A's replay time
    buf = torch.zeros(1 << 20, device="cuda")

    def wall(fn, reps=5):
        sync_all()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync_all()
        return (time.perf_counter() - t0) * 1000 / reps

    run_a(); sync_all()
    only_a = wall(run_a)
    probe = torch.cuda.CUDAGraph()
    with torch.cuda.graph(probe, stream=B):
        o.burn(buf, 2_000_000)
    with torch.cuda.stream(B):
        probe.replay()
    sync_all()
    def run_probe():
        with torch.cuda.stream(B):
            probe.replay()
    t_probe = wall(run_probe)
    iters = max(int(2_000_000 * only_a / max(t_probe, 1e-3)), 1)
    gb = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb, stream=B):
        o.burn(buf, iters)

    def run_b():
        with torch.cuda.stream(B):
            gb.replay()

    run_b(); sync_all()
    only_b = wall(run_b)
    both = wall(lambda: (run_a(), run_b()))
    hidden = (only_a + only_b - both) / only_a
    res.update(only_a_ms=only_a, only_b_ms=only_b, both_ms=both, hidden_fraction_of_A=hidden)

    # event intervals of one concurrent launch (same starter event -> one timeline)
    starter = torch.cuda.Event(enable_timing=True)
    ev = {n: torch.cuda.Event(enable_timing=True) for n in ("a0", "a1", "b0", "b1")}
    sync_all()
    starter.record()
    A.wait_stream(torch.cuda.current_stream()); B.wait_stream(torch.cuda.current_stream())
    ev["a0"].record(A); ev["b0"].record(B)
    run_a(); run_b()
    ev["a1"].record(A); ev["b1"].record(B)
    sync_all()
    t = {n: starter.elapsed_time(e) for n, e in ev.items()}
    inter = max(0.0, min(t["a1"], t["b1"]) - max(t["a0"], t["b0"]))
    res["event_intervals_ms"] = t
    res["event_overlap_fraction_of_A"] = inter / max(t["a1"] - t["a0"], 1e-6)

    res["pass"] = bool(res["bitwise_vs_eager"] and res["partition_kept"] and hidden > 0.5)
    with open(os.path.join(OUT, f"green_{MODE}.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps(res, indent=1, default=str))
    sys.stdout.flush()
    os._exit(0 if res["pass"] else 1)


main()
