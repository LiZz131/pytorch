"""Glue between SGLang and the cuda_graph_memopt package (lives at <repo>/cuda_graph_memopt).

Everything is inert unless the CGMO environment variable selects a feature
(see cuda_graph_memopt/config.py).
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4)))


def _import_package():
    try:
        import cuda_graph_memopt  # noqa: F401
    except ImportError:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        import cuda_graph_memopt  # noqa: F401
    return sys.modules["cuda_graph_memopt"]


def cgmo_requested() -> bool:
    return bool(os.environ.get("CGMO", "").strip()) and os.environ.get("CGMO", "").strip().lower() not in (
        "0", "none", "off", "false",
    )


def uses_vmm_arena() -> bool:
    """True when the tensor feature maps its arena through cuMemMap (custom all-reduce must know)."""
    if not cgmo_requested():
        return False
    return bool(_import_package().vmm_arena_active())


def create_graph_memopt():
    """GraphMemOpt for one CudaGraphRunner, or None when CGMO is not set."""
    if not cgmo_requested():
        return None
    pkg = _import_package()
    return pkg.GraphMemOpt.from_env()
