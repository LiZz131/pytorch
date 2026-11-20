"""Shared helpers of the cuda_graph_memopt GPU tests: CUDA ops, synthetic models, streams.

`Model` mimics what the folder and the planner care about in a DeepSeek decode graph: prologue
kernels, N identical layers (main chain + fork/join side stream, per-layer weight pointers, an
external EVENT_RECORD after every layer like NCCL's serialEvent), epilogue kernels.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline


OUT = os.environ.get("CGMO_TEST_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(OUT, exist_ok=True)

CPP = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

__global__ void layer_kernel(const float* __restrict__ x, const float* __restrict__ W,
                             float* __restrict__ y, int n, int repeats, float bias) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float acc = 0.f;
    for (int rep = 0; rep < repeats; ++rep) {
        acc = bias;
        for (int c = 0; c < n; ++c) acc += x[c] * W[(size_t)row * n + c];
        acc = acc > 0.f ? acc : 0.f;
    }
    y[row] = acc;
}
__global__ void add_kernel(const float* a, const float* b, float* c, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = (a[i] + b[i]) * s;
}
__global__ void mark_kernel(int* seen) {
    unsigned id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    if (threadIdx.x == 0 && id < 256) seen[id] = 1;
}
__global__ void burn_kernel(float* x, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a = (idx < n) ? x[idx] : 1.f;
    for (int i = 0; i < iters; ++i) a = a * 1.0000001f + 0.0000001f;
    if (idx < n) x[idx] = a;
}

void layer(torch::Tensor x, torch::Tensor W, torch::Tensor y, int64_t repeats, double bias) {
    int n = static_cast<int>(W.size(0));
    layer_kernel<<<(n + 31) / 32, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(), n,
        static_cast<int>(repeats), static_cast<float>(bias));
}
// Same kernel, launched through the packed `extra` parameter buffer
// (like DeepGEMM / CuTe JIT launchers), so the graph node has kernelParams == NULL.
void layer_packed(torch::Tensor x, torch::Tensor W, torch::Tensor y, int64_t repeats, double bias) {
    struct Packed { const float* x; const float* W; float* y; int n; int repeats; float bias; };
    Packed pk{x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(),
              static_cast<int>(W.size(0)), static_cast<int>(repeats), static_cast<float>(bias)};
    size_t size = offsetof(Packed, bias) + sizeof(float);  // exact parameter-block size (36 B)
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &pk, CU_LAUNCH_PARAM_BUFFER_SIZE, &size,
                     CU_LAUNCH_PARAM_END};
    cudaFunction_t f = nullptr;
    TORCH_CHECK(cudaGetFuncBySymbol(&f, (const void*)layer_kernel) == cudaSuccess, "func lookup");
    int n = pk.n;
    CUresult r = cuLaunchKernel((CUfunction)f, (n + 31) / 32, 1, 1, 32, 1, 1, 0,
                                (CUstream)at::cuda::getCurrentCUDAStream().stream(), nullptr, extra);
    TORCH_CHECK(r == CUDA_SUCCESS, "cuLaunchKernel(extra) failed ", (int)r);
}

void add2(torch::Tensor a, torch::Tensor b, torch::Tensor c, double s) {
    int n = static_cast<int>(a.numel());
    add_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n, static_cast<float>(s));
}
void mark(torch::Tensor seen) {
    mark_kernel<<<256, 128, 0, at::cuda::getCurrentCUDAStream()>>>(seen.data_ptr<int>());
}
void burn(torch::Tensor x, int64_t iters) {
    burn_kernel<<<512, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), static_cast<int>(x.numel()), static_cast<int>(iters));
}

#define DRV(call) do { CUresult _e = (call); TORCH_CHECK(_e == CUDA_SUCCESS, #call, " failed: ", (int)_e); } while (0)

// Two green contexts (smA, smB); two streams in A (main + fork), one stream in B.
torch::Tensor make_green(int64_t smA, int64_t smB) {
    DRV(cuInit(0));
    CUdevice dev = 0;
    DRV(cuDeviceGet(&dev, 0));
    CUdevResource input{};
    DRV(cuDeviceGetDevResource(dev, &input, CU_DEV_RESOURCE_TYPE_SM));
    CUdevResource res[4]{};
    unsigned groups = 1;
    DRV(cuDevSmResourceSplitByCount(&res[2], &groups, &input, &res[3], 0, (unsigned)(smA + smB)));
    CUdevResourceDesc desc[3]{};
    DRV(cuDevResourceGenerateDesc(&desc[2], &res[2], 1));
    CUgreenCtx scratch{};
    DRV(cuGreenCtxCreate(&scratch, desc[2], dev, CU_GREEN_CTX_DEFAULT_STREAM));
    DRV(cuGreenCtxGetDevResource(scratch, &input, CU_DEV_RESOURCE_TYPE_SM));
    groups = 1;
    // Ask for the B side first; the remainder is A (the driver rounds small counts).
    DRV(cuDevSmResourceSplitByCount(&res[0], &groups, &input, &res[1], 0, (unsigned)smB));
    DRV(cuDevResourceGenerateDesc(&desc[0], &res[1], 1));  // A = remainder
    DRV(cuDevResourceGenerateDesc(&desc[1], &res[0], 1));  // B
    CUgreenCtx g[2]{};
    DRV(cuGreenCtxCreate(&g[0], desc[0], dev, CU_GREEN_CTX_DEFAULT_STREAM));
    DRV(cuGreenCtxCreate(&g[1], desc[1], dev, CU_GREEN_CTX_DEFAULT_STREAM));
    // Same call as sgl-kernel greenctx_stream.cu (CUDA >= 12.5): the stream belongs to the
    // green context directly, which is what pdmux captures on.
    auto mk = [](CUgreenCtx gc) {
        CUstream s{};
        DRV(cuGreenCtxStreamCreate(&s, gc, CU_STREAM_NON_BLOCKING, 0));
        if (getenv("GF_DEBUG_CTX")) {
            CUcontext c1 = nullptr; CUgreenCtx g1 = nullptr;
            CUresult r0 = cuStreamGetCtx_v2(s, &c1, &g1);
            CUcontext c2 = nullptr;
            CUresult r1 = cuCtxFromGreenCtx(&c2, gc);
            CUresult p1 = c1 ? cuCtxPushCurrent(c1) : CUDA_ERROR_INVALID_VALUE;
            if (p1 == CUDA_SUCCESS) cuCtxPopCurrent(nullptr);
            CUresult p2 = cuCtxPushCurrent(c2);
            if (p2 == CUDA_SUCCESS) cuCtxPopCurrent(nullptr);
            fprintf(stderr, "[dbg] streamGetCtx rc=%d ctx=%p green=%p | gc=%p fromGreen(gc)=%p rc=%d | push(streamCtx)=%d push(fromGreen)=%d\n",
                    (int)r0, (void*)c1, (void*)g1, (void*)gc, (void*)c2, (int)r1, (int)p1, (int)p2);
        }
        return (int64_t)(uintptr_t)s;
    };
    auto out = torch::empty({5}, torch::dtype(torch::kInt64).device(torch::kCPU));
    out[0] = mk(g[0]);
    out[1] = mk(g[0]);
    out[2] = mk(g[1]);
    out[3] = (int64_t)res[1].sm.smCount;
    out[4] = (int64_t)res[0].sm.smCount;
    DRV(cuGreenCtxDestroy(scratch));
    return out;
}
"""

DECL = """
#include <torch/extension.h>
void layer(torch::Tensor x, torch::Tensor W, torch::Tensor y, int64_t repeats, double bias);
void add2(torch::Tensor a, torch::Tensor b, torch::Tensor c, double s);
void layer_packed(torch::Tensor x, torch::Tensor W, torch::Tensor y, int64_t repeats, double bias);
void mark(torch::Tensor seen);
void burn(torch::Tensor x, int64_t iters);
torch::Tensor make_green(int64_t smA, int64_t smB);
"""

N = 2048  # vector width == rows of each per-layer weight matrix
REPEATS = 4

_ops = None


def ops():
    global _ops
    if _ops is None:
        _ops = load_inline(
            name="cgmo_unit_ops2",
            cpp_sources=[DECL],
            cuda_sources=[CPP],
            functions=["layer", "layer_packed", "add2", "mark", "burn", "make_green"],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
            verbose=False,
        )
    return _ops


_cudart = None


def cudart():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so.12")
    return _cudart


class Model:
    """prologue(2) + L layers + epilogue(2); per layer: main kernel, side kernel
    (fork/join), add, external EVENT_RECORD, optional SM marker."""

    def __init__(self, layers, seen=None):
        self.L = layers
        self.packed = os.environ.get("GF_EXTRA", "0") == "1"
        self.seen = seen
        g = torch.Generator(device="cuda").manual_seed(1234)
        self.W = [torch.randn(N, N, device="cuda", generator=g) * (1.0 / N**0.5) for _ in range(layers)]
        self.V = [torch.randn(N, N, device="cuda", generator=g) * (1.0 / N**0.5) for _ in range(layers)]
        self.Wout = torch.randn(N, N, device="cuda", generator=g) * (1.0 / N**0.5)
        self.x = torch.randn(N, device="cuda", generator=g)
        self.ext_ev = torch.cuda.Event()
        self.ext_ev.record()  # create the cudaEvent_t
        torch.cuda.synchronize()
        self.out = None

    def ext_record(self, stream):
        # external flag is only legal while capturing (like NCCL's strongstream)
        flags = 1 if torch.cuda.is_current_stream_capturing() else 0
        r = cudart().cudaEventRecordWithFlags(
            ctypes.c_void_p(self.ext_ev.cuda_event), ctypes.c_void_p(stream.cuda_stream), flags
        )
        assert r == 0, f"cudaEventRecordWithFlags external -> {r}"

    def ext_wait(self, stream):
        flags = 1 if torch.cuda.is_current_stream_capturing() else 0
        r = cudart().cudaStreamWaitEvent(
            ctypes.c_void_p(stream.cuda_stream), ctypes.c_void_p(self.ext_ev.cuda_event), flags
        )
        assert r == 0, f"cudaStreamWaitEvent external -> {r}"

    def forward(self, side):
        o = ops()
        main = torch.cuda.current_stream()
        self.ext_wait(main)
        cur = torch.empty_like(self.x)
        o.add2(self.x, self.x, cur, 0.5)  # prologue 1
        t = torch.empty_like(self.x)
        o.layer(cur, self.Wout, t, 1, 0.0)  # prologue 2
        cur = t
        for l in range(self.L):
            fork = torch.cuda.Event()
            fork.record(main)
            with torch.cuda.stream(side):
                side.wait_event(fork)
                z = torch.empty_like(cur)
                (o.layer_packed if self.packed else o.layer)(cur, self.V[l], z, REPEATS, 0.01)
                join = torch.cuda.Event()
                join.record(side)
            y = torch.empty_like(cur)
            o.layer(cur, self.W[l], y, REPEATS, 0.02)
            main.wait_event(join)
            nxt = torch.empty_like(cur)
            o.add2(y, z, nxt, 0.5)
            if self.seen is not None:
                o.mark(self.seen)
            self.ext_record(main)
            cur = nxt
        e1 = torch.empty_like(cur)
        o.layer(cur, self.Wout, e1, 1, 0.0)  # epilogue 1
        e2 = torch.empty_like(cur)
        o.add2(e1, cur, e2, 1.0)  # epilogue 2
        self.out = e2


def bitwise_equal(a, b):
    """Bit pattern equality (NaN == NaN), independent of float semantics."""
    return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def diff_stats(a, b):
    d = (a - b).abs()
    return {
        "n_mismatch_bits": int((a.view(torch.int32) != b.view(torch.int32)).sum()),
        "max_abs_diff": float(torch.nan_to_num(d, nan=-1.0).max()),
        "ref_finite": bool(torch.isfinite(a).all()),
        "got_finite": bool(torch.isfinite(b).all()),
        "ref_absmax": float(torch.nan_to_num(a.abs(), nan=-1.0).max()),
    }


def used_mib():
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 2**20



# ---------------------------------------------------------------------------------------------
# additional models and helpers for the cuda_graph_memopt tests
# ---------------------------------------------------------------------------------------------

PKG_ROOT = os.environ.get("CGMO_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


class MemcpyModel(Model):
    """Model whose layers contain a device-to-device cudaMemcpy node: the folder must refuse it."""

    def forward(self, side):
        o = ops()
        main = torch.cuda.current_stream()
        cur = torch.empty_like(self.x)
        o.add2(self.x, self.x, cur, 0.5)
        for l in range(self.L):
            y = torch.empty_like(cur)
            o.layer(cur, self.W[l], y, REPEATS, 0.02)
            c2 = torch.empty_like(cur)
            c2.copy_(y)  # contiguous D2D copy -> MEMCPY graph node
            nxt = torch.empty_like(cur)
            o.add2(c2, cur, nxt, 0.5)
            cur = nxt
        self.out = cur


class ScratchModel(Model):
    """Model + per-layer scratch tensors of 3/5/2/7 MiB whose lifetimes span two layers."""

    SIZES = [int(x) for x in os.environ.get("SCR_SIZES", "3,5,2,7").split(",")]

    def forward(self, side):
        o = ops()
        main = torch.cuda.current_stream()
        self.ext_wait(main)
        cur = torch.empty_like(self.x)
        o.add2(self.x, self.x, cur, 0.5)
        t = torch.empty_like(self.x)
        o.layer(cur, self.Wout, t, 1, 0.0)
        cur = t
        prev = None
        for l in range(self.L):
            fork = torch.cuda.Event()
            fork.record(main)
            s = torch.empty(self.SIZES[l % len(self.SIZES)] * (1 << 18), device="cuda")
            if not os.environ.get("SCR_NOFILL"):
                s.fill_(float(l % 5))
            with torch.cuda.stream(side):
                side.wait_event(fork)
                z = torch.empty_like(cur)
                o.layer(cur, self.V[l], z, REPEATS, 0.01)
                join = torch.cuda.Event()
                join.record(side)
            y = torch.empty_like(cur)
            o.layer(cur, self.W[l], y, REPEATS, 0.02)
            main.wait_event(join)
            nxt = torch.empty_like(cur)
            o.add2(y, z, nxt, 0.5)
            if os.environ.get("SCR_NOBIAS"):
                nxt2 = nxt
            else:
                bias = s[: N] * 1e-3
                if prev is not None:
                    bias = bias + prev[: N] * 1e-4
                nxt2 = nxt + bias
            if self.seen is not None:
                o.mark(self.seen)
            self.ext_record(main)
            cur, prev = nxt2, s
        e1 = torch.empty_like(cur)
        o.layer(cur, self.Wout, e1, 1, 0.0)
        e2 = torch.empty_like(cur)
        o.add2(e1, cur, e2, 1.0)
        self.out = e2


class FragModel:
    """Allocation pattern that is hostile to online first-fit: sizes of 1..34 MiB, random lifetimes.

    Every tensor is written by a kernel, the observable result is a 1024-float accumulator.
    Kernel geometry depends on the tensor size, so the graph has no periodic region (fold refuses).
    """

    SIZES_MIB = [1, 2, 3, 5, 8, 13, 21, 34]

    def __init__(self, n_ops=120, seed=0, scale=1):
        import random

        rng = random.Random(seed)
        self.steps = []  # ("alloc", id, mib) / ("free", id)
        live = []
        for i in range(n_ops):
            if live and rng.random() < 0.45:
                self.steps.append(("free", live.pop(rng.randrange(len(live)))))
            self.steps.append(("alloc", i, rng.choice(self.SIZES_MIB) * scale))
            live.append(i)
        g = torch.Generator(device="cuda").manual_seed(seed + 99)
        self.base = torch.randn(max(self.SIZES_MIB) * scale * (1 << 18), device="cuda", generator=g)
        self.acc = torch.zeros(1024, device="cuda")
        self.out = None

    def forward(self, side=None):
        acc = torch.zeros(1024, device="cuda")
        live = {}
        for kind, i, *rest in self.steps:
            if kind == "alloc":
                numel = rest[0] * (1 << 18)
                t = self.base[:numel].clone()
                t.mul_(1.0 + 1e-3 * (i % 7))
                acc.add_(t[:1024] * 0.25)
                live[i] = t
            else:
                del live[i]
        for t in live.values():
            acc.add_(t[:1024] * 0.125)
        self.out = acc


class GrowModel:
    """Worst case of an online allocator: every tensor is larger than the one freed before it.

    Tensor k has (base + k) MiB (all >= 10 MiB, so each gets its own exactly-sized segment), and two
    tensors are live at a time. The caching allocator cannot reuse the freed block for the next, bigger
    request, so it reserves the sum of all sizes; the peak live size is only the two largest.
    """

    def __init__(self, k=40, base=12, seed=0):
        self.sizes = [base + i for i in range(k)]
        g = torch.Generator(device="cuda").manual_seed(seed + 5)
        self.base = torch.randn(max(self.sizes) * (1 << 18), device="cuda", generator=g)
        self.out = None

    def forward(self, side=None):
        acc = torch.zeros(1024, device="cuda")
        prev = None
        for mib in self.sizes:
            t = self.base[: mib * (1 << 18)].clone()
            t.mul_(1.001)
            acc.add_(t[:1024] * 0.5)
            if prev is not None:
                acc.add_(prev[:1024] * 0.25)
            prev = t
        self.out = acc


class NondetModel(Model):
    """The first call inside a capture allocates one more tensor than the second: plan mismatch."""

    def __init__(self, layers):
        super().__init__(layers)
        self.calls = 0

    def forward(self, side):
        in_capture = torch.cuda.is_current_stream_capturing()
        if in_capture:
            self.calls += 1
        extra = torch.empty(1 << 20, device="cuda") if (in_capture and self.calls == 1) else None
        if extra is not None:
            extra.fill_(1.0)
        super().forward(side)


def driver_used_mib() -> float:
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 2**20


def make_streams(use_green: bool):
    """Returns (main, side, other) streams. With green contexts: main+side in A (36 SM), other in B (96 SM)."""
    if not use_green:
        return torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    ptrs = ops().make_green(36, 96).tolist()
    streams = [torch.cuda.ExternalStream(p) for p in ptrs[:3]]
    return streams[0], streams[1], streams[2], ptrs[3], ptrs[4]


def sync_streams(*streams):
    # torch.cuda.synchronize() does not wait for work on green-context streams
    for s in streams:
        s.synchronize()
    torch.cuda.synchronize()


def replay_ms(opt, key, graph, stream, reps=100):
    """Mean GPU time of one replay on `stream` (events), through the manager when given."""
    def once():
        with torch.cuda.stream(stream):
            if opt is not None:
                opt.replay(key, graph)
            else:
                graph.replay()

    for _ in range(5):
        once()
    stream.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        e0.record(stream)
    for _ in range(reps):
        once()
    with torch.cuda.stream(stream):
        e1.record(stream)
    stream.synchronize()
    return e0.elapsed_time(e1) / reps


def warm(model, main, side, n=2):
    for _ in range(n):
        with torch.cuda.stream(main):
            model.forward(side)
        sync_streams(main, side)
    model.out = None


def capture_model(model, main, side, opt, key="k", group=0, pool=None, do_warm=True):
    """Warm up and capture `model.forward(side)` on `main`; returns (graph, out).

    opt=None is the stock path (torch.cuda.graph); otherwise GraphMemOpt.capture. Mirrors SGLang:
    two warm-up runs, then one capture.
    """
    if do_warm:
        warm(model, main, side)
    if pool is None:
        pool = torch.cuda.graph_pool_handle()
    def fn():
        # Like SGLang's run_once: return the outputs, keep no reference on any long-lived object
        # (a stale reference to a record-pass tensor would pin the record pass' memory).
        model.out = None
        model.forward(side)
        out, model.out = model.out, None
        return out

    # Cost of the capture = driver memory taken after the allocator cache of the warm-up is returned.
    sync_streams(main, side)
    torch.cuda.empty_cache()
    u0, r0 = driver_used_mib(), torch.cuda.memory_reserved() / 2**20
    if opt is None:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=main):
            out = fn()
    else:
        graph = opt.new_graph()
        out = opt.capture(graph, pool, main, fn, key=key, group=group)
    sync_streams(main, side)
    model.out = out
    model.cap_stats = {"driver_delta_mib": driver_used_mib() - u0,
                       "reserved_delta_mib": torch.cuda.memory_reserved() / 2**20 - r0}
    return graph, out
