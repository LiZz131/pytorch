#!/usr/bin/env python3
"""GPU unit tests of cuda_graph_memopt (one case in one mode per process).

    python3 test_gpu_unit.py <case> <mode>

cases : chain      160-layer decode-like graph (fork/join, external events)      fold shows here
        scratch    same, plus per-layer scratch tensors of varying size/lifetime  tensor + fold
        frag       random-size, random-lifetime allocation pattern                tensor, typical gain
        grow       each tensor larger than the one freed before it                tensor, online-allocator worst case
        multi      three graphs sharing one pool (the SGLang pattern)             reuse-pool path
        multigroup 5 graphs, each on its own stream of one stream mutex group,     pdmux pattern: later captures
                   one shared pool, growing-size allocation pattern                 are planned into the arena
        memcpy     layers contain MEMCPY nodes: fold must refuse and fall back
        nondet     fn allocates differently on its second call: plan must be rejected, fall back
        strict     memcpy with CGMO_STRICT=1: must raise instead of falling back
        verify     chain with CGMO_VERIFY=1: manager's own bitwise check of folded vs unrolled
        dump       scratch with CGMO_DEBUG=2: checks the dump directory contents
modes : off | tensor | fold | both     (mode sets CGMO; "off" runs the stock torch.cuda.graph path)

Writes results/unit/<case>_<mode>.json and .pt (outputs, for cross-mode bitwise comparison).
Run via run_unit.sh, summarised by summarize_unit.py.
"""

from __future__ import annotations

import json
import os
import sys

CASE, MODE = sys.argv[1], sys.argv[2]
os.environ.pop("CGMO", None)
if MODE != "off":
    os.environ["CGMO"] = MODE if MODE != "both" else "tensor,fold"
if CASE == "strict":
    os.environ["CGMO_STRICT"] = "1"
if CASE == "verify":
    os.environ["CGMO_VERIFY"] = "1"
if CASE == "dump":
    os.environ["CGMO_DEBUG"] = "2"
    os.environ["CGMO_DUMP_DIR"] = os.path.join(os.environ.get("CGMO_TEST_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")), "unit", f"dump_{MODE}")
    os.environ["CGMO_DUMP_MAX"] = "2"

import torch  # noqa: E402

import cgmo_testlib as T  # noqa: E402
from cuda_graph_memopt import GraphMemOpt  # noqa: E402

OUT = os.path.join(T.OUT, "unit")
os.makedirs(OUT, exist_ok=True)
LAYERS = int(os.environ.get("GF_LAYERS", "160"))


ScratchModel = T.ScratchModel


def build():
    if CASE in ("chain", "verify"):
        return T.Model(LAYERS)
    if CASE in ("scratch", "dump"):
        return ScratchModel(LAYERS)
    if CASE == "grow":
        return T.GrowModel()
    if CASE == "frag":
        return T.FragModel(n_ops=160, seed=3)
    if CASE in ("memcpy", "strict"):
        return T.MemcpyModel(LAYERS)
    if CASE == "nondet":
        return T.NondetModel(LAYERS)
    raise SystemExit(f"unknown case {CASE}")


def set_input(model, trial):
    torch.cuda.synchronize()
    g = torch.Generator(device="cuda").manual_seed(100 + trial)
    if hasattr(model, "x"):
        model.x.copy_(torch.randn(model.x.numel(), device="cuda", generator=g) * (1 + trial))
    else:  # FragModel: the input is the base tensor
        model.base.copy_(torch.randn(model.base.numel(), device="cuda", generator=g))
    torch.cuda.synchronize()  # the copy runs on the default stream; the graph runs on another one


def run_multi(opt, res):
    """Three FragModel graphs of different size in one pool: every replay must equal its eager run."""
    main, side, _ = T.make_streams(False)
    pool = torch.cuda.graph_pool_handle()
    models = [T.FragModel(n_ops=80, seed=s, scale=sc) for s, sc in ((1, 2), (2, 1), (3, 1))]
    for m in models:
        T.warm(m, main, side)
    torch.cuda.empty_cache()
    u0, r0 = T.driver_used_mib(), torch.cuda.memory_reserved() / 2**20
    graphs = []
    for i, m in enumerate(models):
        graphs.append(T.capture_model(m, main, side, opt, key=f"m{i}", group=0, pool=pool, do_warm=False))
    torch.cuda.synchronize()
    res["driver_delta_mib"] = T.driver_used_mib() - u0
    res["reserved_delta_mib"] = torch.cuda.memory_reserved() / 2**20 - r0
    outs, ok = [], True
    for i, (m, (g, out)) in enumerate(zip(models, graphs)):
        for trial in range(2):
            set_input(m, trial)
            with torch.cuda.stream(main):
                m.forward(side)
            T.sync_streams(main, side)
            ref = m.out.clone()
            with torch.cuda.stream(main):
                opt.replay(f"m{i}", g) if opt else g.replay()
            T.sync_streams(main, side)
            ok &= T.bitwise_equal(ref, out)
            outs.append(out.clone().cpu())
    res["bitwise_vs_eager"] = bool(ok)
    res["replay_ms"] = sum(T.replay_ms(opt, f"m{i}", g, main) for i, (m, (g, out)) in enumerate(zip(models, graphs))) / 3
    torch.save(outs, os.path.join(OUT, f"{CASE}_{MODE}.pt"))


def run_multigroup(opt, res):
    """The pdmux pattern: one pool, every graph captured on a different stream of one stream mutex group.

    Online first-fit needs a multiple of the live peak for this allocation pattern; the stream mutex
    group lets the later captures reuse those blocks. Without planning *into the arena* the tensor
    feature would reserve a second full-size segment at the second stream (that is what the first
    SGLang integration run showed: pool 3706 MiB with and without the feature).
    """
    n = 5
    ks = [40, 40, 32, 40, 20]  # 'batch sizes': the same allocation pattern, different length
    streams = [(torch.cuda.Stream(), torch.cuda.Stream()) for _ in range(n)]
    reg = getattr(torch._C, "_cuda_registerStreamMutexGroup", None)
    res["smg_registered"] = reg is not None
    if reg is not None:
        reg([m.cuda_stream for m, _ in streams], "cgmo_test_group")
    pool = torch.cuda.graph_pool_handle()
    models = [T.GrowModel(k=k) for k in ks]
    for m, (main_s, side_s) in zip(models, streams):
        T.warm(m, main_s, side_s)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    u0, r0 = T.driver_used_mib(), torch.cuda.memory_reserved() / 2**20
    graphs = []
    for i, (m, (main_s, side_s)) in enumerate(zip(models, streams)):
        graphs.append(T.capture_model(m, main_s, side_s, opt, key=f"g{i}", group=i, pool=pool, do_warm=False))
    torch.cuda.synchronize()
    res["driver_delta_mib"] = T.driver_used_mib() - u0
    res["reserved_delta_mib"] = torch.cuda.memory_reserved() / 2**20 - r0
    outs, ok = [], True
    for i, (m, (main_s, side_s), (g, out)) in enumerate(zip(models, streams, graphs)):
        for trial in range(2):
            set_input(m, trial)
            with torch.cuda.stream(main_s):
                m.forward(side_s)
            T.sync_streams(main_s, side_s)
            ref = m.out.clone()
            with torch.cuda.stream(main_s):
                opt.replay(f"g{i}", g) if opt else g.replay()
            T.sync_streams(main_s, side_s)
            ok &= T.bitwise_equal(ref, out)
            outs.append(out.clone().cpu())
    res["bitwise_vs_eager"] = bool(ok)
    res["replay_ms"] = sum(T.replay_ms(opt, f"g{i}", g, st[0]) for i, (st, (g, out)) in enumerate(zip(streams, graphs))) / n
    torch.save(outs, os.path.join(OUT, f"{CASE}_{MODE}.pt"))


def main():
    res = {"case": CASE, "mode": MODE, "pass": False}
    opt = GraphMemOpt.from_env() if MODE != "off" else None
    if CASE == "multi":
        run_multi(opt, res)
    elif CASE == "multigroup":
        run_multigroup(opt, res)
    else:
        mainS, side, _ = T.make_streams(False)
        model = build()
        raised = None
        try:
            graph, out = T.capture_model(model, mainS, side, opt, key="g", group=0)
        except Exception as exc:  # strict case expects this
            raised = f"{type(exc).__name__}: {exc}"
        res["raised"] = raised
        if raised is None:
            res.update(model.cap_stats)
            outs, ok = [], True
            if opt is not None:
                opt.verify_pending()
            for trial in range(3):
                set_input(model, trial)
                with torch.cuda.stream(mainS):
                    model.forward(side)
                T.sync_streams(mainS, side)
                ref = model.out.clone()
                with torch.cuda.stream(mainS):
                    opt.replay("g", graph) if opt else graph.replay()
                T.sync_streams(mainS, side)
                ok &= T.bitwise_equal(ref, out) and bool(torch.isfinite(out).all())
                outs.append(out.clone().cpu())
            res["bitwise_vs_eager"] = bool(ok)
            res["replay_ms"] = T.replay_ms(opt, "g", graph, mainS)
            torch.save(outs, os.path.join(OUT, f"{CASE}_{MODE}.pt"))
    if opt is not None:
        res["summary"] = opt.finish()
    # ------------------------------------------------------------- per-case verdicts
    if CASE == "strict":
        res["pass"] = (MODE == "off") or (MODE in ("fold", "both") and res["raised"] is not None and "refused" in res["raised"]) \
            or (MODE == "tensor" and res["raised"] is None)
    elif CASE == "nondet":
        # tensor: plan mismatch must be detected, graph must still be correct (fallback)
        tensor_modes = ("tensor", "both")
        fell_back = opt is None or MODE not in tensor_modes or any(
            r.get("tensor", {}).get("mode") == "fallback" for r in opt.records)
        res["fell_back"] = fell_back
        res["pass"] = res.get("bitwise_vs_eager", False) and fell_back
    elif CASE == "memcpy":
        refused = opt is None or MODE not in ("fold", "both") or bool(opt.refused)
        res["fold_refused"] = refused
        res["pass"] = res.get("bitwise_vs_eager", False) and refused
    elif CASE == "verify":
        v = (opt.verify_results if opt else [])
        res["verify"] = v
        need = MODE in ("fold", "both")
        res["pass"] = res.get("bitwise_vs_eager", False) and (not need or (len(v) >= 1 and all(x["fold_vs_orig_bitwise"] for x in v)))
    elif CASE == "dump":
        res["dump_check"] = check_dump(res) if MODE != "off" else {}
        res["pass"] = res.get("bitwise_vs_eager", False) and (MODE == "off" or res["dump_check"].get("ok", False))
    else:
        res["pass"] = bool(res.get("bitwise_vs_eager", False))
    path = os.path.join(OUT, f"{CASE}_{MODE}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "summary"}, indent=1, default=str))
    if opt is not None:
        opt.close()
    sys.stdout.flush()
    if os.environ.get("CGMO_NORMAL_EXIT"):  # nsys needs the normal exit path to flush its buffers
        torch.cuda.synchronize()
        sys.exit(0 if res["pass"] else 1)
    os._exit(0 if res["pass"] else 1)


def check_dump(res):
    """Everything the debug mode promises must exist and parse."""
    root = os.environ["CGMO_DUMP_DIR"]
    found = {"files": [], "problems": []}
    caps = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            found["files"].append(os.path.relpath(os.path.join(dp, fn), root))
        if os.path.basename(dp).startswith("capture_"):
            caps.append(dp)
    if not caps:
        found["problems"].append("no capture_* directory")
    want = ["meta.json", "report.md"]
    if MODE in ("tensor", "both"):
        want += ["tensor/trace.jsonl", "tensor/buffers.csv", "tensor/plan.json", "tensor/layout.svg", "tensor/timeline.txt", "tensor/segments_before.json"]
    if MODE in ("fold", "both"):
        want += ["fold/graph_before.json", "fold/fold_classes.json", "fold/fold_analysis.json",
                 "fold/graph_after_parent.json", "fold/graph_after_body.json", "fold/fold_summary.md",
                 "fold/graph_before.dot", "fold/graph_after.dot"]
    for c in caps:
        for w in want:
            p = os.path.join(c, w)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                found["problems"].append(f"missing/empty {os.path.relpath(p, root)}")
            elif w.endswith(".json"):
                try:
                    json.load(open(p))
                except ValueError as exc:
                    found["problems"].append(f"bad json {w}: {exc}")
        if MODE in ("fold", "both") and os.path.exists(os.path.join(c, "fold/graph_before.json")):
            gb = json.load(open(os.path.join(c, "fold/graph_before.json")))
            ga = json.load(open(os.path.join(c, "fold/graph_after_parent.json")))
            body = json.load(open(os.path.join(c, "fold/graph_after_body.json")))
            found["nodes_before"], found["nodes_after_parent"], found["nodes_body"] = gb["n_nodes"], ga["n_nodes"], body["n_nodes"]
            if not ga["n_nodes"] < gb["n_nodes"]:
                found["problems"].append("folded parent graph is not smaller")
            if not any(n.get("demangled") for n in gb["nodes"] if n["type"] == "KERNEL"):
                found["problems"].append("kernel names not demangled")
            an = json.load(open(os.path.join(c, "fold/fold_analysis.json")))
            found["region"] = an["region"]
        if MODE in ("tensor", "both") and os.path.exists(os.path.join(c, "tensor/plan.json")):
            pl = json.load(open(os.path.join(c, "tensor/plan.json")))
            found["tensor_plan"] = {k: pl.get(k) for k in ("n_buffers", "peak_live_bytes", "packed_bytes", "arena_bytes", "record_segment_bytes", "verify")}
            svg = open(os.path.join(c, "tensor/layout.svg")).read()
            import xml.dom.minidom as md

            try:
                md.parseString(svg)
            except Exception as exc:
                found["problems"].append(f"layout.svg not valid XML: {exc}")
    found["captures"] = len(caps)
    found["ok"] = not found["problems"]
    return found


if __name__ == "__main__":
    main()
