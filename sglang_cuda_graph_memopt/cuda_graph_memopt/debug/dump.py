"""Human-oriented dumps of what cuda_graph_memopt analysed and decided (debug level 2).

One directory per dumped capture::

    capture_0003_g0_128/
      meta.json               key, group, config, timing, mode of each feature
      report.md               the story of this capture, tensor and fold, before vs after
      tensor/
        trace.jsonl           allocator events of the record pass (alloc / free_completed, addr, size)
        buffers.csv           one row per buffer: size, lifetime [start,end), planned offset, address
                              the caching allocator gave it in the record pass
        segments_before.json  cudaMalloc segments of the record pass
        plan.json             arena / peak-live / packed numbers and verified invariants
        layout.svg            lifetime x address plot, before (record pass) and after (plan)
        timeline.txt          live bytes over allocation ticks (ASCII)
      fold/
        graph_before.json     every node of the captured graph (kernel symbol, geometry, param hash)
        fold_classes.json     launch classes, run-length-encoded class sequence, period candidates
        fold_analysis.json    chosen region, body positions, per-iteration parameter updates
        graph_after_parent.json / graph_after_body.json
        graph_before.dot / graph_after.dot   Graphviz of the collapsed structure
        fold_summary.md       readable digest of the three json files above
"""

from __future__ import annotations

import bisect
import csv
import ctypes
import json
import os
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

_demangle_cache: dict[str, str] = {}
_cxa = None


def demangle(sym: str) -> str:
    """C++ demangle via libstdc++'s __cxa_demangle; the mangled name when that fails."""
    global _cxa
    if not sym:
        return sym
    if sym in _demangle_cache:
        return _demangle_cache[sym]
    out = sym
    try:
        if _cxa is None:
            libcxx = ctypes.CDLL("libstdc++.so.6")
            libcxx.__cxa_demangle.restype = ctypes.c_void_p
            libcxx.__cxa_demangle.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            libc = ctypes.CDLL(None)
            libc.free.argtypes = [ctypes.c_void_p]
            _cxa = (libcxx, libc)
        libcxx, libc = _cxa
        status = ctypes.c_int(0)
        p = libcxx.__cxa_demangle(sym.encode(), None, None, ctypes.byref(status))
        if p and status.value == 0:
            out = ctypes.string_at(p).decode(errors="replace")
        if p:
            libc.free(p)
    except Exception:
        pass
    _demangle_cache[sym] = out
    return out


def short_name(sym: str, width: int = 60) -> str:
    """Demangled kernel name without the template argument lists, truncated."""
    name = demangle(sym)
    depth, buf = 0, []
    for ch in name:
        if ch == "<":
            depth += 1
            if depth == 1:
                buf.append("<..>")
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            buf.append(ch)
    s = "".join(buf)
    s = re.sub(r"\(.*\)$", "", s)
    s = re.sub(r"^void ", "", s)
    return s if len(s) <= width else s[: width - 1] + "…"


def mib(n: Optional[float]) -> str:
    return "n/a" if n is None else f"{n / (1 << 20):.2f} MiB"


def _write(path: str, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text)


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# tensor
# ---------------------------------------------------------------------------------------------


def _buffer_addresses(res) -> list[tuple[int, int]]:
    """(segment index, offset inside segment) of every buffer in the record pass."""
    bases = [s["address"] for s in res.segments]
    out = []
    for e in (e for e in res.events if e["action"] == "alloc"):
        i = bisect.bisect_right(bases, e["addr"]) - 1
        out.append((i, e["addr"] - bases[i]) if i >= 0 else (-1, e["addr"]))
    return out


def _live_curve(buffers) -> list[int]:
    n_ticks = max([b.start for b in buffers] + [b.end or 0 for b in buffers] + [0]) + 2
    delta = [0] * (n_ticks + 1)
    for b in buffers:
        delta[b.start] += b.size
        if b.end is not None:
            delta[b.end] -= b.size
    cur, curve = 0, []
    for d in delta[:n_ticks]:
        cur += d
        curve.append(cur)
    return curve


def _layout_svg(res, addrs, path: str, title: str) -> None:
    """Two panels with a shared scale: left = record pass, right = plan."""
    buffers = res.buffers
    if not buffers:
        return
    ticks = max([b.start for b in buffers] + [b.end or 0 for b in buffers]) + 1
    seg_base, run = [], 0
    for s in res.segments:
        seg_base.append(run)
        run += s["size"]
    before_total = max(run, 1)
    after_total = max(res.arena_bytes, 1)
    ymax = max(before_total, after_total)
    W, H, M = 520, 420, 48
    total_w = 2 * W + 3 * M
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#9c755f"]

    def rect(x0, x1, y0, h, col, persistent):
        # y grows downward in SVG; address 0 at the bottom.
        px = M + x0 / ticks * W
        pw = max((x1 - x0) / ticks * W, 0.4)
        py = M + H - (y0 + h) / ymax * H
        ph = max(h / ymax * H, 0.3)
        stroke = ' stroke="#000" stroke-width="0.3"' if persistent else ""
        return f'<rect x="{px:.2f}" y="{py:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="{col}" fill-opacity="0.75"{stroke}/>'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{H + 2 * M + 24}" font-family="monospace" font-size="11">',
        f'<text x="{M}" y="16" font-size="13">{title}</text>',
    ]
    for panel, (name, total) in enumerate((("before: record pass (caching allocator)", before_total),
                                           ("after: greedy-by-size plan (arena)", after_total))):
        ox = panel * (W + M)
        parts.append(f'<g transform="translate({ox},0)">')
        parts.append(f'<text x="{M}" y="{M - 8}">{name}  total={mib(total)}</text>')
        parts.append(f'<rect x="{M}" y="{M}" width="{W}" height="{H}" fill="none" stroke="#999"/>')
        parts.append(f'<line x1="{M}" x2="{M + W}" y1="{M + H - total / ymax * H:.1f}" y2="{M + H - total / ymax * H:.1f}" stroke="#c00" stroke-dasharray="4 3"/>')
        for k in range(5):
            v = ymax * k / 4
            y = M + H - v / ymax * H
            parts.append(f'<text x="{M - 4}" y="{y + 4:.1f}" text-anchor="end">{v / (1 << 20):.0f}</text>')
        parts.append(f'<text x="{M + W / 2}" y="{M + H + 18}" text-anchor="middle">allocation tick (0 .. {ticks})   y: MiB</text>')
        for b in buffers:
            end = b.end if b.end is not None else ticks
            col = palette[b.index % len(palette)]
            if panel == 0:
                si, off = addrs[b.index] if b.index < len(addrs) else (-1, 0)
                y0 = (seg_base[si] + off) if si >= 0 else 0
            else:
                y0 = b.offset
            parts.append(rect(b.start, end, y0, b.size, col, b.end is None))
        if panel == 0:
            for sb in seg_base[1:]:
                y = M + H - sb / ymax * H
                parts.append(f'<line x1="{M}" x2="{M + W}" y1="{y:.1f}" y2="{y:.1f}" stroke="#555" stroke-width="0.5"/>')
        parts.append("</g>")
    parts.append(f'<text x="{M}" y="{H + 2 * M + 12}">black outline: buffer still live at the end of the capture (address baked into outputs); red dashed: total footprint</text>')
    parts.append("</svg>")
    _write(path, "\n".join(parts))


def dump_tensor(res, directory: str) -> dict:
    """Write the tensor artifacts of one capture; returns the numbers used by report.md."""
    os.makedirs(directory, exist_ok=True)
    info: dict[str, Any] = {"mode": res.mode, **res.report}
    if res.events:
        with open(os.path.join(directory, "trace.jsonl"), "w") as fh:
            for i, e in enumerate(res.events):
                fh.write(json.dumps({"i": i, **e, "addr": hex(e["addr"])}) + "\n")
    if res.segments:
        _write(os.path.join(directory, "segments_before.json"), json.dumps(res.segments, indent=1))
    if res.mode in ("packed", "skipped_no_gain") and res.buffers:
        from ..tensor import plan as planmod

        addrs = _buffer_addresses(res)
        with open(os.path.join(directory, "buffers.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "size", "start", "end", "planned_offset", "planned_end",
                        "record_segment", "record_segment_offset"])
            for b in res.buffers:
                seg, off = addrs[b.index] if b.index < len(addrs) else (-1, -1)
                w.writerow([b.index, b.size, b.start, "" if b.end is None else b.end, b.offset,
                            b.offset + b.size, seg, off])
        ver = planmod.verify_plan(res.buffers, res.arena_bytes)
        curve = _live_curve(res.buffers)
        peak_tick = max(range(len(curve)), key=curve.__getitem__)
        persistent = [b for b in res.buffers if b.end is None]
        info.update(
            verify=ver,
            peak_tick=peak_tick,
            persistent_buffers=len(persistent),
            persistent_bytes=sum(b.size for b in persistent),
            record_total_segments=len(res.segments),
        )
        _write(os.path.join(directory, "plan.json"), json.dumps(info, indent=1, default=str))
        _layout_svg(res, addrs, os.path.join(directory, "layout.svg"),
                    f"{info.get('n_buffers')} buffers, peak live {mib(info.get('peak_live_bytes'))}")
        # ASCII live-bytes timeline, 64 columns
        cols, rows = 64, 10
        bucket = max(1, len(curve) // cols)
        col_max = [max(curve[i:i + bucket]) for i in range(0, len(curve), bucket)]
        top = max(col_max) or 1
        lines = [f"live bytes vs allocation tick; peak {mib(top)} at tick {peak_tick}; "
                 f"arena {mib(res.arena_bytes)}; one column = {bucket} ticks"]
        for r in range(rows, 0, -1):
            thr = top * r / rows
            lines.append(f"{thr / (1 << 20):9.1f} |" + "".join("#" if v >= thr else " " for v in col_max))
        lines.append(" " * 10 + "+" + "-" * len(col_max))
        _write(os.path.join(directory, "timeline.txt"), "\n".join(lines) + "\n")
        top_n = sorted(res.buffers, key=lambda b: -b.size)[:10]
        info["largest"] = [
            {"index": b.index, "size": b.size, "life": [b.start, b.end], "offset": b.offset} for b in top_n
        ]
    else:
        _write(os.path.join(directory, "plan.json"), json.dumps(info, indent=1, default=str))
    return info


# ---------------------------------------------------------------------------------------------
# fold
# ---------------------------------------------------------------------------------------------


def _dot(graph: dict, path: str, title: str, max_nodes: int = 400) -> None:
    """Graphviz of a dumped graph; consecutive kernels with equal symbol/geometry are not merged,
    but graphs above ``max_nodes`` nodes are cut to the first/last halves with a marker node."""
    nodes, edges = graph["nodes"], graph["edges"]
    keep = {n["id"] for n in nodes}
    if len(nodes) > max_nodes:
        half = max_nodes // 2
        ids = [n["id"] for n in nodes]
        keep = set(ids[:half]) | set(ids[-half:])
    lines = [f'digraph G {{ rankdir=TB; label="{title} ({graph["n_nodes"]} nodes, {graph["n_edges"]} edges)"; '
             'node [shape=box, fontsize=9, fontname=monospace];']
    for n in nodes:
        if n["id"] not in keep:
            continue
        label = n["type"]
        color = {"KERNEL": "white", "EVENT_RECORD": "#ffe9a8", "EVENT_WAIT": "#ffd0a8",
                 "EMPTY": "#eeeeee", "CONDITIONAL": "#a8d8ff"}.get(n["type"], "#dddddd")
        if n["type"] == "KERNEL":
            label = f'{n["id"]}: {short_name(n.get("symbol", ""), 36)}\\n{n.get("grid")}x{n.get("block")}'
        else:
            label = f'{n["id"]}: {n["type"]}'
        lines.append(f'  n{n["id"]} [label="{label}", style=filled, fillcolor="{color}"];')
    for a, b, *_ in edges:
        if a in keep and b in keep:
            lines.append(f"  n{a} -> n{b};")
    if len(keep) < len(nodes):
        lines.append('  cut [label="... nodes omitted (see json) ...", shape=note];')
    lines.append("}")
    _write(path, "\n".join(lines) + "\n")


def dump_fold_post(directory: str, report: dict, status: str) -> dict:
    """Post-process the JSON the C++ side wrote: demangle, digest, Graphviz. Returns digest numbers."""
    digest: dict[str, Any] = {"status": status, "driver_report": report}
    before = _load(os.path.join(directory, "graph_before.json"))
    classes = _load(os.path.join(directory, "fold_classes.json"))
    analysis = _load(os.path.join(directory, "fold_analysis.json"))
    parent = _load(os.path.join(directory, "graph_after_parent.json"))
    body = _load(os.path.join(directory, "graph_after_body.json"))
    for doc in (before, parent, body):
        if doc:
            for n in doc["nodes"]:
                if "symbol" in n:
                    n["demangled"] = demangle(n["symbol"])
    for doc, name in ((before, "graph_before.json"), (parent, "graph_after_parent.json"), (body, "graph_after_body.json")):
        if doc:
            _write(os.path.join(directory, name), json.dumps(doc, indent=0))
    if before:
        _dot(before, os.path.join(directory, "graph_before.dot"), "before fold")
    if parent:
        _dot(parent, os.path.join(directory, "graph_after.dot"), "after fold (parent)")
    if body:
        _dot(body, os.path.join(directory, "graph_after_body.dot"), "after fold (WHILE body)")

    md: list[str] = [f"# fold digest ({status})", ""]
    r = report
    md.append(f"- driver message: `{r.get('message', '')}`")
    md.append(f"- nodes: {r.get('n_nodes')} total, {r.get('n_kernels')} kernels, {r.get('n_edges')} edges "
              f"(event record {r.get('n_event_record')}, event wait {r.get('n_event_wait')})")
    md.append(f"- launch classes: {r.get('n_classes')} (kernels with equal function, geometry, smem, parameter layout, launch attrs)")
    if classes:
        md += ["", "## launch classes", "", "| class | count | kernel | grid | block | smem | param bytes |", "|---|---|---|---|---|---|---|"]
        for c in classes["classes"]:
            md.append(f"| {c['id']} | {c['count']} | `{short_name(c['symbol'], 70)}` | {c['grid']} | {c['block']} | {c['smem']} | {c['param_bytes']} |")
        md += ["", "## class sequence (run-length encoded, node creation order)", "",
               "`" + " ".join(f"{c}x{n}" if n > 1 else str(c) for c, n in classes["sequence_rle"][:400]) + "`", ""]
        cands = sorted(classes["period_candidates"], key=lambda c: -c["cover"])[:8]
        md += ["## period candidates (best periodic run per period length P, by kernels covered)", "",
               "| P | start kernel | repeats | kernels covered |", "|---|---|---|---|"]
        for c in cands:
            md.append(f"| {c['P']} | {c['start']} | {c['iters']} | {c['cover']} |")
        md.append("")
    if analysis:
        reg = analysis["region"]
        md += ["## chosen region", "",
               f"- period P = {reg['period']} kernels, repeated {reg['iters']} times, kernels [{reg['start_kernel']}, {reg['end_kernel_exclusive']})",
               f"- prefix kernels {reg['prefix_kernels']}, suffix kernels {reg['suffix_kernels']}",
               f"- EVENT_RECORD / EMPTY nodes inside the region removed: {reg['loop_events_removed']}",
               f"- dependencies entering the loop (nodes before it): {len(analysis['in_deps_gidx'])}; leaving: {len(analysis['out_succ_gidx'])}; "
               f"cross-iteration edges: {analysis['cross_iteration_edges']}; edges re-added by event bypass: {analysis['bypass_edges_added']}",
               f"- body positions and intra-iteration edges: {len(analysis['body_edges'])}", ""]
        md += ["### body (iteration 0)", "", "| pos | class | kernel | grid | block | param bytes |", "|---|---|---|---|---|---|"]
        for p in analysis["body_positions"]:
            md.append(f"| {p['pos']} | {p['class']} | `{short_name(p['symbol'], 60)}` | {p['grid']} | {p['block']} | {p['param_bytes']} |")
        md += ["", f"### per-iteration parameter updates: {len(analysis['updates'])} spans, table {analysis['table']['row_bytes']} B/iteration x {analysis['table']['iters']} = {analysis['table']['total_bytes']} B", ""]
        md += ["| span | body pos | param offset | bytes | iter0 | iter1 | iter2 | last |", "|---|---|---|---|---|---|---|---|"]
        for i, u in enumerate(analysis["updates"][:40]):
            v = u["values"]
            md.append(f"| {i} | {u['pos']} | {u['offset']} | {u['size']} | `{v[0][:16]}` | `{v[1][:16] if len(v) > 1 else ''}` | `{v[2][:16] if len(v) > 2 else ''}` | `{v[-1][:16]}` |")
        if len(analysis["updates"]) > 40:
            md.append(f"| ... | | | | | | | ({len(analysis['updates']) - 40} more spans in fold_analysis.json) |")
        md.append("")
    if status == "folded":
        md += ["## result", "",
               f"- parent graph: {r.get('n_nodes')} -> {r.get('folded_nodes_actual')} nodes; WHILE body: {r.get('body_nodes')} nodes (iteration 0 + scheduler)",
               f"- PDL (programmatic) launch attributes turned into full edges: {r.get('pdl_attr_dropped')}",
               f"- driver memory taken by instantiating the folded graph: {mib(r.get('instantiate_driver_delta'))}",
               f"- context mode: {r.get('ctx_mode')} (0 current, 1 stream's (green) context pushed, 2 node's context pushed)", ""]
    _write(os.path.join(directory, "fold_summary.md"), "\n".join(md) + "\n")
    digest["classes"] = classes and {"n": len(classes["classes"])}
    return digest


# ---------------------------------------------------------------------------------------------
# per-capture report
# ---------------------------------------------------------------------------------------------


def write_report(directory: str, meta: dict, tensor: Optional[dict], fold: Optional[dict]) -> None:
    _write(os.path.join(directory, "meta.json"), json.dumps(meta, indent=1, default=str))
    md = [f"# capture {meta['key']} (group {meta.get('group')}, seq {meta['seq']})", "",
          f"- features: {meta['features']}   wall time {meta['wall_ms']:.0f} ms",
          f"- stream {meta.get('stream')}  pool {meta.get('pool')}", ""]
    if tensor is not None:
        md += ["## tensor placement", "", f"mode: **{tensor['mode']}**" + (
            " (the arena would not be smaller than what the caching allocator took: stock capture kept)"
            if tensor["mode"] == "skipped_no_gain" else ""), ""]
        if tensor["mode"] in ("packed", "skipped_no_gain"):
            rb = tensor.get("record_segment_bytes")
            md += ["| quantity | value |", "|---|---|",
                   f"| buffers | {tensor['n_buffers']} ({tensor.get('persistent_buffers')} persistent, {mib(tensor.get('persistent_bytes'))}) |",
                   f"| allocator events | {tensor['n_events']} |",
                   f"| peak live bytes (lower bound for any placement) | {mib(tensor['peak_live_bytes'])} at tick {tensor.get('peak_tick')} |",
                   f"| before: segments the caching allocator reserved (record pass) | {mib(rb)} in {tensor.get('record_total_segments')} segments |",
                   f"| before: reserved delta of the record pass | {mib(tensor['record_reserved_delta'])} |",
                   f"| after: packed span (max offset+size) | {mib(tensor['packed_bytes'])} |",
                   f"| after: arena (one segment) | {mib(tensor['arena_bytes'])} |",
                   f"| after: physical bytes mapped (VMM) | {mib(tensor.get('physical_bytes'))} |",
                   f"| after: reserved delta of the packed pass | {mib(tensor.get('packed_reserved_delta'))} |",
                   f"| packed / peak-live | {tensor['packed_bytes'] / max(tensor['peak_live_bytes'], 1):.3f} |",
                   f"| invariants | {tensor.get('verify')} |", ""]
            md += ["`tensor/layout.svg` shows both layouts; `tensor/buffers.csv` has every buffer; "
                   "`tensor/timeline.txt` the live-bytes curve.", "", "Largest buffers:", "",
                   "| index | size | lifetime [start,end) | planned offset |", "|---|---|---|---|"]
            for b in tensor.get("largest", []):
                md.append(f"| {b['index']} | {mib(b['size'])} | {b['life']} | {b['offset']} |")
            md.append("")
        else:
            md.append(f"`{json.dumps(tensor, default=str)}`")
            md.append("")
    if fold is not None:
        md += ["## fold", "", f"status: **{fold['status']}**", "", "See `fold/fold_summary.md` for the full digest.", ""]
        r = fold["driver_report"]
        md += ["| quantity | before | after |", "|---|---|---|",
               f"| graph nodes | {r.get('n_nodes')} | {r.get('folded_nodes_actual') or 'n/a'} (+{r.get('body_nodes')} in WHILE body) |",
               f"| region | | period {r.get('period')} x {r.get('iters')} iterations |",
               f"| executable memory | measured by the manager (see meta.json: driver_delta) | |", ""]
    _write(os.path.join(directory, "report.md"), "\n".join(md) + "\n")
