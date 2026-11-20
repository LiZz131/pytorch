"""ctypes front end of graph_fold.cu (fold repeated layers of a captured graph into a WHILE node).

    lib = GraphFoldLib.get(lib_dir)
    folded, info = lib.fold(torch_cuda_graph.raw_cuda_graph(), stream_ptr, dump_dir=None)
    folded.launch(torch.cuda.current_stream().cuda_stream)

The shared object needs the CUDA device runtime (device-side cudaGraphKernelNodeSetParam), so it
is built with ``-rdc=true`` and a device link step. ``build.sh fold`` prebuilds it into
``<pkg>/lib``; if it is missing, the first use builds it under a file lock.
"""

from __future__ import annotations

import ctypes
import fcntl
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("cuda_graph_memopt")

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIELDS = (
    "status n_nodes n_kernels n_edges n_event_record n_event_wait "
    "n_nondefault_edges n_classes region_start period iters prefix_kernels "
    "suffix_kernels loop_events in_edges out_edges cross_edges body_edges "
    "n_updates row_bytes table_bytes unrolled_nodes folded_nodes "
    "folded_nodes_actual body_nodes instantiate_driver_delta pdl_attr_dropped ctx_mode"
).split()


class GfReport(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int64) for n in _FIELDS] + [("message", ctypes.c_char * 512)]

    def as_dict(self) -> dict:
        d = {n: getattr(self, n) for n in _FIELDS}
        d["message"] = self.message.decode(errors="replace")
        return d


def build(so_path: str) -> None:
    out_dir = os.path.dirname(so_path)
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(_HERE, "graph_fold.cu")
    obj = os.path.join(out_dir, "graph_fold.o")
    dlink = os.path.join(out_dir, "graph_fold_dlink.o")
    common = ["nvcc", "-O3", "-std=c++17", "-arch=native", "-I", _HERE, "-Xcompiler", "-fPIC"]
    for cmd in (
        common + ["-rdc=true", "-c", src, "-o", obj],
        common + ["-dlink", obj, "-o", dlink, "-lcudadevrt"],
        common + ["-shared", obj, dlink, "-o", so_path, "-lcudadevrt", "-lcuda"],
    ):
        logger.info("graph_fold build: %s", " ".join(cmd))
        subprocess.check_call(cmd)


@dataclass
class FoldedGraph:
    lib: "GraphFoldLib"
    handle: ctypes.c_void_p
    report: dict

    def launch(self, stream_ptr: int) -> None:
        err = self.lib.so.gf_launch(self.handle, ctypes.c_void_p(stream_ptr))
        if err != 0:
            raise RuntimeError(f"gf_launch failed with CUresult {err}")

    def state(self) -> dict:
        out = (ctypes.c_int64 * 3)()
        self.lib.so.gf_state(self.handle, out)
        return {"idx": out[0], "sched_error": out[1], "trips": out[2]}

    def close(self) -> None:
        if self.handle:
            self.lib.so.gf_destroy(self.handle)
            self.handle = None


class GraphFoldLib:
    _inst: Optional["GraphFoldLib"] = None

    def __init__(self, so_path: str):
        self.path = so_path
        self.so = ctypes.CDLL(so_path)
        self.so.gf_fold.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ctypes.POINTER(GfReport), ctypes.POINTER(ctypes.c_void_p),
        ]
        self.so.gf_fold.restype = ctypes.c_int
        self.so.gf_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.so.gf_launch.restype = ctypes.c_int
        self.so.gf_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
        self.so.gf_state.restype = ctypes.c_int
        self.so.gf_destroy.argtypes = [ctypes.c_void_p]
        self.so.gf_destroy.restype = None
        self.so.gf_dump_graph.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self.so.gf_dump_graph.restype = ctypes.c_int

    @classmethod
    def get(cls, lib_dir: str) -> "GraphFoldLib":
        if cls._inst is None:
            so = os.path.join(lib_dir, "graph_fold.so")
            os.makedirs(lib_dir, exist_ok=True)
            with open(os.path.join(lib_dir, ".fold.lock"), "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    srcs = [os.path.join(_HERE, "graph_fold.cu"), os.path.join(_HERE, "graph_fold.h")]
                    if (not os.path.exists(so)) or os.path.getmtime(so) < max(map(os.path.getmtime, srcs)):
                        build(so)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
            cls._inst = cls(so)
            logger.info("fold: graph_fold.so=%s", so)
        return cls._inst

    def analyze(self, raw_graph: int, min_iters: int = 4, min_saved_nodes: int = 0,
                dump_dir: Optional[str] = None) -> dict:
        """Dry run: find the periodic region and report; the graph is not modified."""
        rep = GfReport()
        self.so.gf_fold(ctypes.c_void_p(raw_graph), None, min_iters, min_saved_nodes, 1,
                        dump_dir.encode() if dump_dir else None, ctypes.byref(rep), None)
        return rep.as_dict()

    def fold(self, raw_graph: int, stream_ptr: int = 0, min_iters: int = 4, min_saved_nodes: int = 0,
             dump_dir: Optional[str] = None) -> tuple[Optional[FoldedGraph], dict]:
        """``stream_ptr``: the stream the graph was captured on (selects the Green Context)."""
        rep = GfReport()
        handle = ctypes.c_void_p()
        rc = self.so.gf_fold(
            ctypes.c_void_p(raw_graph), ctypes.c_void_p(stream_ptr or 0), min_iters, min_saved_nodes, 0,
            dump_dir.encode() if dump_dir else None, ctypes.byref(rep), ctypes.byref(handle),
        )
        info = rep.as_dict()
        if rc != 0 or not handle.value:
            return None, info
        return FoldedGraph(self, handle, info), info

    def dump_graph(self, raw_graph: int, path: str, label: str = "") -> bool:
        return self.so.gf_dump_graph(ctypes.c_void_p(raw_graph), path.encode(), label.encode()) == 0
