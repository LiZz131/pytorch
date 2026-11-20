"""Configuration of cuda_graph_memopt, read from environment variables.

Environment (all optional; with nothing set the tool is fully inactive):

    CGMO=<features>        comma list of: tensor, fold, stats, all (= tensor,fold), none   (default none)
                           stats: no optimization, only the debug bookkeeping (baseline for comparisons)
    CGMO_TENSOR_SLACK=f    tensor: skip packing only if the arena exceeds first-fit's reserve by more
                           than this fraction (default 0.05)
    CGMO_TENSOR_VMM=0|1    tensor: reserve the arena as VMM VA and map physical pages
                           after planning (default 1)
    CGMO_FOLD_MIN_ITERS=N  fold: minimum number of repeats of the periodic region (default 4)
    CGMO_FOLD_MIN_SAVED_NODES=N  fold: skip graphs where the fold removes fewer nodes (default 256,
                           about 1 MiB of executable-graph memory)
    CGMO_VERIFY=N          fold: replay-compare the first N folded graphs per group (default 0)
    CGMO_STRICT=0|1        raise instead of falling back when a feature cannot be applied
    CGMO_DEBUG=0|1|2       0 off; 1 log + per-capture records; 2 also dump files (see below)
    CGMO_DUMP_DIR=path     where level-2 dumps go (default ./cgmo_dump)
    CGMO_DUMP_MAX=N        level 2: dump at most N captures per process (default 4)
    CGMO_DUMP_FILTER=re    level 2: only captures whose key matches (default: the first
                           capture of every stream group, i.e. the largest batch size)
    CGMO_DUMP_RANKS=0,1|all  level 2: ranks that dump (default 0)
    CGMO_LIB_DIR=path      directory with libc10_cuda.so / graph_fold.so (default <pkg>/lib)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

# "stats" optimizes nothing: it only enables the bookkeeping (pool statistics in captures.jsonl, dumps),
# so a baseline run can be compared with an optimized one on exactly the same numbers.
_OPTIMIZATIONS = ("tensor", "fold")
_FEATURES = _OPTIMIZATIONS + ("stats",)


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def parse_features(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return ()
    out: list[str] = []
    for tok in spec.replace("+", ",").split(","):
        tok = tok.strip().lower()
        if tok in ("", "none", "0", "off", "false"):
            continue
        if tok in ("all", "1", "on", "true"):
            out.extend(_OPTIMIZATIONS)
        elif tok in _FEATURES:
            out.append(tok)
        else:
            raise ValueError(f"CGMO: unknown feature {tok!r}; valid: {_FEATURES + ('all', 'none')}")
    return tuple(f for f in _FEATURES if f in out)


@dataclass(frozen=True)
class Config:
    features: tuple[str, ...] = ()
    tensor_vmm: bool = True
    tensor_slack: float = 0.05
    fold_min_iters: int = 4
    fold_min_saved_nodes: int = 256
    verify: int = 0
    strict: bool = False
    debug: int = 0
    dump_dir: str = "cgmo_dump"
    dump_max: int = 4
    dump_filter: str = ""
    dump_ranks: str = "0"
    lib_dir: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        here = os.path.dirname(os.path.abspath(__file__))
        return cls(
            features=parse_features(os.environ.get("CGMO")),
            tensor_vmm=_flag("CGMO_TENSOR_VMM", True),
            tensor_slack=float(os.environ.get("CGMO_TENSOR_SLACK", "0.05")),
            fold_min_iters=_int("CGMO_FOLD_MIN_ITERS", 4),
            fold_min_saved_nodes=_int("CGMO_FOLD_MIN_SAVED_NODES", 256),
            verify=_int("CGMO_VERIFY", 0),
            strict=_flag("CGMO_STRICT", False),
            debug=_int("CGMO_DEBUG", 0),
            dump_dir=os.environ.get("CGMO_DUMP_DIR", "cgmo_dump"),
            dump_max=_int("CGMO_DUMP_MAX", 4),
            dump_filter=os.environ.get("CGMO_DUMP_FILTER", ""),
            dump_ranks=os.environ.get("CGMO_DUMP_RANKS", "0"),
            lib_dir=os.environ.get("CGMO_LIB_DIR", os.path.join(here, "lib")),
        )

    @property
    def tensor(self) -> bool:
        return "tensor" in self.features

    @property
    def fold(self) -> bool:
        return "fold" in self.features

    @property
    def enabled(self) -> bool:
        return bool(self.features)

    @property
    def uses_vmm(self) -> bool:
        """True when the tensor arena is mapped through cuMemMap (affects custom all-reduce)."""
        return self.tensor and self.tensor_vmm

    def as_dict(self) -> dict:
        d = asdict(self)
        d["features"] = list(self.features)
        return d


def vmm_arena_active() -> bool:
    """Cheap, torch-free check used by custom_all_reduce at import/construction time."""
    try:
        return Config.from_env().uses_vmm
    except ValueError:
        return False
