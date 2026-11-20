"""cuda_graph_memopt: reduce the GPU memory of captured CUDA graphs.

Two independent features, selected with the CGMO environment variable (see config.py):

    tensor  place the intermediate tensors of a capture with a static plan instead of the
            caching allocator's online first-fit  -> smaller graph memory pool
    fold    replace the unrolled repeated layers of a captured graph with one WHILE node
            -> smaller executable graph (cudaGraphExec_t) memory

Both keep the numerical behaviour of the graph and work on Green Context streams.
Documentation: cuda-graph-memopt-context.md next to this package.
"""

from .config import Config, vmm_arena_active

__all__ = ["Config", "GraphMemOpt", "vmm_arena_active"]


def __getattr__(name):  # torch is only imported when the manager is actually used
    if name == "GraphMemOpt":
        from .manager import GraphMemOpt

        return GraphMemOpt
    raise AttributeError(name)
