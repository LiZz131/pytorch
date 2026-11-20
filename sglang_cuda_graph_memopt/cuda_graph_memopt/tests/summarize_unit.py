#!/usr/bin/env python3
"""Cross-mode summary of the GPU unit tests -> results/unit_summary.json and results/unit_summary.md.

Criteria checked here (in addition to each process' own verdict):
  * numerics     : every mode's replay outputs are bitwise equal to mode `off` (same inputs)
  * memory       : chain/scratch/multi/frag: capture cost (driver memory) of the optimised modes < off
  * performance  : replay time of each mode within PERF_TOL of off
"""

from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("CGMO_TEST_OUT", os.path.join(HERE, "results"))
U = os.path.join(RES, "unit")
PERF_TOL = float(os.environ.get("PERF_TOL", "1.15"))
MODES = ["off", "tensor", "fold", "both"]
CASES = ["chain", "scratch", "frag", "grow", "multi", "multigroup", "memcpy", "nondet", "strict", "verify", "dump"]


def load(case, mode):
    p = os.path.join(U, f"{case}_{mode}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def outs(case, mode):
    p = os.path.join(U, f"{case}_{mode}.pt")
    return torch.load(p) if os.path.exists(p) else None


def same_outputs(a, b):
    return a is not None and b is not None and len(a) == len(b) and all(
        torch.equal(x.contiguous().view(torch.int32), y.contiguous().view(torch.int32)) for x, y in zip(a, b))


def main():
    rows, verdicts = [], []
    for case in CASES:
        ref = load(case, "off")
        ref_out = outs(case, "off")
        for mode in MODES:
            r = load(case, mode)
            if r is None:
                continue
            row = {"case": case, "mode": mode, "pass_process": r.get("pass"),
                   "driver_delta_mib": r.get("driver_delta_mib"), "reserved_delta_mib": r.get("reserved_delta_mib"),
                   "replay_ms": r.get("replay_ms"), "bitwise_vs_eager": r.get("bitwise_vs_eager")}
            if ref and mode != "off" and ref.get("replay_ms") and r.get("replay_ms"):
                row["perf_ratio"] = r["replay_ms"] / ref["replay_ms"]
            if ref_out is not None and mode != "off" and outs(case, mode) is not None:
                row["bitwise_vs_off"] = same_outputs(ref_out, outs(case, mode))
            if ref and mode != "off" and ref.get("driver_delta_mib") is not None and r.get("driver_delta_mib") is not None:
                row["driver_saved_mib"] = ref["driver_delta_mib"] - r["driver_delta_mib"]
                row["driver_saved_pct"] = 100.0 * row["driver_saved_mib"] / max(ref["driver_delta_mib"], 1e-9)
            s = r.get("summary") or {}
            if s.get("tensor"):
                row["tensor_modes"] = s["tensor"]["modes"]
                row["arena_mib"] = s["tensor"]["arena_bytes_total"] / 2**20
                row["peak_live_mib"] = s["tensor"]["peak_live_bytes_total"] / 2**20
            if s.get("fold"):
                row["folded"] = s["fold"]["folded_graphs"]
                row["refused"] = s["fold"]["refused_graphs"]
                row["nodes"] = f'{s["fold"]["unrolled_nodes_total"]} -> {s["fold"]["folded_parent_nodes_total"]}'
            rows.append(row)

    def get(case, mode):
        return next((r for r in rows if r["case"] == case and r["mode"] == mode), None)

    def verdict(name, ok, detail=""):
        verdicts.append({"check": name, "pass": bool(ok), "detail": detail})

    for r in rows:
        verdict(f"{r['case']}/{r['mode']}: process verdict", r["pass_process"])
        if r.get("bitwise_vs_off") is not None:
            verdict(f"{r['case']}/{r['mode']}: outputs bitwise equal to off", r["bitwise_vs_off"])
        if r.get("perf_ratio") is not None and r["case"] in ("chain", "scratch", "frag", "grow", "multi", "multigroup"):
            verdict(f"{r['case']}/{r['mode']}: replay time within {int((PERF_TOL - 1) * 100)}% of off", r["perf_ratio"] <= PERF_TOL,
                    f"ratio {r['perf_ratio']:.3f}")
    # memory claims
    for case, mode in (("chain", "fold"), ("chain", "both"), ("scratch", "fold"), ("scratch", "both"),
                       ("frag", "tensor"), ("frag", "both"), ("grow", "tensor"), ("grow", "both"), ("scratch", "tensor"), ("multi", "tensor"), ("multigroup", "tensor"), ("multigroup", "both")):
        r = get(case, mode)
        if r and r.get("driver_saved_mib") is not None:
            verdict(f"{case}/{mode}: capture memory below off", r["driver_saved_mib"] > 0,
                    f"saved {r['driver_saved_mib']:.1f} MiB ({r['driver_saved_pct']:.1f}%)")

    md = ["# GPU unit test matrix", "",
          "| case | mode | pass | driver delta (MiB) | saved vs off | reserved delta (MiB) | replay ms | perf vs off | bitwise vs off | notes |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        note = []
        if r.get("tensor_modes"):
            note.append(f"tensor {r['tensor_modes']}, arena {r['arena_mib']:.1f} MiB, peak live {r['peak_live_mib']:.1f} MiB")
        if r.get("nodes"):
            note.append(f"fold folded={r['folded']} refused={r['refused']} nodes {r['nodes']}")
        f = lambda v, fmt="{:.1f}": "" if v is None else fmt.format(v)
        md.append(f"| {r['case']} | {r['mode']} | {'PASS' if r['pass_process'] else 'FAIL'} | {f(r['driver_delta_mib'])} | "
                  f"{f(r.get('driver_saved_mib'))} ({f(r.get('driver_saved_pct'))}%) | {f(r['reserved_delta_mib'])} | {f(r['replay_ms'], '{:.2f}')} | "
                  f"{f(r.get('perf_ratio'), '{:.3f}')} | {r.get('bitwise_vs_off', '')} | {'; '.join(note)} |")
    md += ["", "## checks", "", "| check | result | detail |", "|---|---|---|"]
    for v in verdicts:
        md.append(f"| {v['check']} | {'PASS' if v['pass'] else 'FAIL'} | {v['detail']} |")
    bad = [v for v in verdicts if not v["pass"]]
    md += ["", f"**{len(verdicts) - len(bad)}/{len(verdicts)} checks passed**"]
    open(os.path.join(RES, "unit_summary.md"), "w").write("\n".join(md) + "\n")
    json.dump({"rows": rows, "verdicts": verdicts}, open(os.path.join(RES, "unit_summary.json"), "w"), indent=1)
    print("\n".join(md))
    sys.exit(1 if bad else 0)


main()
