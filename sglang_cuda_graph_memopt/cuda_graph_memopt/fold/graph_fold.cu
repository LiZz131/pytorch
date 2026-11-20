// graph_fold.cu -- fold the repeated layers of a captured CUDA graph into one
// WHILE conditional node.
//
// Input : a captured, not yet instantiated cudaGraph_t (torch CUDAGraph with
//         keep_graph=True gives one through raw_cuda_graph()).
// Output: an instantiated cudaGraphExec_t whose node count no longer grows with
//         the number of layers.
//
// Algorithm (details and pseudocode in bench_out/graph_memory_opt/GRAPH_FOLD.md):
//   1. read every kernel node (driver API: function, ctx, launch geometry,
//      packed parameter bytes, launch attributes);
//   2. find the longest region [S, S+iters*P) of the kernel sequence, in node
//      creation order, whose launch classes repeat with period P;
//   3. clone the graph; delete the region (kernels + NCCL EVENT_RECORD nodes
//      inside it); add WHILE(body = iteration 0 of the region + scheduler);
//   4. per iteration the scheduler rewrites the bytes of the parameters that
//      differ between iterations (per-layer weight pointers, tensor maps, NCCL
//      work descriptors) from a device table, then sets the loop condition.
// The WHILE node and its body keep the context (Green Context) of the captured
// kernels, so the SM partition is preserved.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "graph_fold.h"

namespace {

struct Fail : std::runtime_error {
    explicit Fail(const std::string& m) : std::runtime_error(m) {}
};

#define DRV(call)                                                                       \
    do {                                                                                \
        CUresult _r = (call);                                                           \
        if (_r != CUDA_SUCCESS) {                                                       \
            const char* _n = nullptr;                                                   \
            cuGetErrorName(_r, &_n);                                                    \
            throw Fail(std::string(#call) + " -> " + (_n ? _n : "?") + " @line " +      \
                       std::to_string(__LINE__));                                       \
        }                                                                               \
    } while (0)

#define RT(call)                                                                        \
    do {                                                                                \
        cudaError_t _e = (call);                                                        \
        if (_e != cudaSuccess) {                                                        \
            throw Fail(std::string(#call) + " -> " + cudaGetErrorName(_e) + " @line " + \
                       std::to_string(__LINE__));                                       \
        }                                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------

struct GfUpd {
    cudaGraphDeviceNode_t node;
    uint32_t offset;     // byte offset inside the kernel parameter buffer
    uint32_t size;       // bytes to rewrite
    uint64_t table_off;  // byte offset inside one table row
};

__global__ void gf_reset(int* idx, int* status, int* trips) {
    *idx = 0;
    *status = 0;
    *trips = 0;
}

// Last node of the WHILE body. Iteration `idx` has completed. Install the
// parameters of iteration idx+1 (or re-arm iteration 0 after the last one) and
// publish the loop condition. All threads of the block share the SetParam work.
__global__ void gf_sched(cudaGraphConditionalHandle handle, const GfUpd* upd, int n_upd,
                         const unsigned char* table, unsigned long long row_bytes, int n_iters,
                         int* idx, int* status, int* trips) {
    __shared__ int s_row;
    __shared__ int s_continue;
    if (threadIdx.x == 0) {
        *trips += 1;
        int next = *idx + 1;
        s_continue = next < n_iters;
        s_row = s_continue ? next : 0;  // after the last trip rewind to iteration 0
    }
    __syncthreads();
    const unsigned char* row = table + static_cast<unsigned long long>(s_row) * row_bytes;
    for (int i = threadIdx.x; i < n_upd; i += blockDim.x) {
        cudaError_t err = cudaGraphKernelNodeSetParam(upd[i].node, upd[i].offset,
                                                      row + upd[i].table_off, upd[i].size);
        if (err != cudaSuccess) atomicMax(status, 1000 + static_cast<int>(err));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *idx = s_row;
        cudaGraphSetConditional(handle, s_continue ? 1u : 0u);
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

constexpr int kMaxAttrId = 12;  // CU_LAUNCH_ATTRIBUTE_* 1..12 are launch-shape attributes

struct KInfo {
    CUgraphNode node{};
    int gidx = 0;  // index in cuGraphGetNodes order
    CUDA_KERNEL_NODE_PARAMS p{};
    std::vector<std::pair<size_t, size_t>> pinfo;  // (offset, size) per parameter
    std::vector<unsigned char> blob;               // packed parameter buffer (padded to 16 B)
    size_t param_bytes = 0;                        // real size of the parameter buffer
    bool use_extra = false;                        // launched through the packed `extra` buffer
    std::vector<unsigned char> attrs;              // concatenated attribute values
    int cls = -1;                                  // launch class id
};

bool same_launch(const KInfo& a, const KInfo& b) {
    return a.p.func == b.p.func && a.p.kern == b.p.kern && a.p.ctx == b.p.ctx &&
           a.p.gridDimX == b.p.gridDimX && a.p.gridDimY == b.p.gridDimY &&
           a.p.gridDimZ == b.p.gridDimZ && a.p.blockDimX == b.p.blockDimX &&
           a.p.blockDimY == b.p.blockDimY && a.p.blockDimZ == b.p.blockDimZ &&
           a.p.sharedMemBytes == b.p.sharedMemBytes && a.pinfo == b.pinfo && a.attrs == b.attrs;
}

KInfo read_kernel(CUgraphNode node, int gidx) {
    KInfo k;
    k.node = node;
    k.gidx = gidx;
    std::memset(&k.p, 0, sizeof(k.p));
    DRV(cuGraphKernelNodeGetParams(node, &k.p));
    if (k.p.func == nullptr && k.p.kern == nullptr) throw Fail("kernel node without func/kern");
    if (k.p.kernelParams == nullptr) {
        // Packed launch: extra = {BUFFER_POINTER, buf, BUFFER_SIZE, &size, END}. The whole
        // buffer is the parameter block, so it is one pseudo-parameter at offset 0.
        if (k.p.extra == nullptr) throw Fail("kernel node has neither kernelParams nor extra");
        void* buf = nullptr;
        size_t size = 0;
        for (void** e = k.p.extra; *e != CU_LAUNCH_PARAM_END; e += 2) {
            if (*e == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = e[1];
            else if (*e == CU_LAUNCH_PARAM_BUFFER_SIZE) size = *reinterpret_cast<size_t*>(e[1]);
            else throw Fail("unknown tag in kernel `extra`");
        }
        if (!buf || !size) throw Fail("kernel `extra` lacks buffer pointer or size");
        k.use_extra = true;
        k.param_bytes = size;
        k.pinfo.emplace_back(0, size);
        k.blob.assign((size + 15) & ~size_t(15), 0);
        std::memcpy(k.blob.data(), buf, size);
    } else {
        for (size_t i = 0;; ++i) {
            size_t off = 0, sz = 0;
            CUresult r = k.p.func ? cuFuncGetParamInfo(k.p.func, i, &off, &sz)
                                  : cuKernelGetParamInfo(k.p.kern, i, &off, &sz);
            if (r != CUDA_SUCCESS) break;  // index past the last parameter
            k.pinfo.emplace_back(off, sz);
            if (i > 4096) throw Fail("implausible parameter count");
        }
        if (k.pinfo.empty()) throw Fail("could not query parameter layout");
        size_t total = 0;
        for (auto& pr : k.pinfo) total = std::max(total, pr.first + pr.second);
        k.param_bytes = total;
        k.blob.assign((total + 15) & ~size_t(15), 0);
        for (size_t i = 0; i < k.pinfo.size(); ++i) {
            std::memcpy(k.blob.data() + k.pinfo[i].first, k.p.kernelParams[i], k.pinfo[i].second);
        }
    }
    for (int id = 1; id <= kMaxAttrId; ++id) {
        CUkernelNodeAttrValue v;
        std::memset(&v, 0, sizeof(v));
        CUresult r = cuGraphKernelNodeGetAttribute(node, static_cast<CUkernelNodeAttrID>(id), &v);
        unsigned char tag = (r == CUDA_SUCCESS) ? 1 : 0;
        k.attrs.push_back(tag);
        if (tag) {
            const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
            k.attrs.insert(k.attrs.end(), b, b + sizeof(v));
        }
    }
    return k;
}

struct Region {
    int S = 0, P = 0, iters = 0;  // kernel-sequence coordinates
};

// One row per candidate period P that has a periodic run of >= min_iters repeats:
// the best run for that P. Filled only for the debug dump.
struct PeriodCandidate {
    int P, S, iters, cover;
};

Region find_region(const std::vector<int>& cls, int min_iters,
                   std::vector<PeriodCandidate>* trace = nullptr) {
    const int n = static_cast<int>(cls.size());
    Region best;
    int best_cover = 0;
    for (int P = 1; P <= n / std::max(min_iters, 2); ++P) {
        int run = 0, run_start = 0;
        PeriodCandidate cand{P, 0, 0, 0};
        for (int i = 0; i + P < n; ++i) {
            if (cls[i] == cls[i + P]) {
                if (run == 0) run_start = i;
                ++run;
                int cover = ((run + P) / P) * P;
                if ((run + P) / P >= min_iters && cover > cand.cover) {
                    cand = {P, run_start, (run + P) / P, cover};
                }
                if (cover > best_cover && (run + P) / P >= min_iters) {
                    best_cover = cover;
                    best = {run_start, P, (run + P) / P};
                }
            } else {
                run = 0;
            }
        }
        if (trace && cand.cover > 0) trace->push_back(cand);
    }
    return best;
}

// ---------------------------------------------------------------------------
// Debug dump (JSON). Nothing in here influences the fold.
// ---------------------------------------------------------------------------

std::string jesc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char t[8];
                    std::snprintf(t, sizeof(t), "\\u%04x", c);
                    o += t;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    return o;
}

std::string hex_bytes(const unsigned char* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string o;
    o.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        o += d[b[i] >> 4];
        o += d[b[i] & 15];
    }
    return o;
}

uint64_t fnv1a(const unsigned char* b, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::string hex64(uint64_t v) {
    char t[24];
    std::snprintf(t, sizeof(t), "%016llx", static_cast<unsigned long long>(v));
    return t;
}

// Mangled symbol of a kernel node (CUDA >= 12.3 driver API); "" when unavailable.
std::string kernel_symbol(const CUDA_KERNEL_NODE_PARAMS& p) {
    const char* name = nullptr;
    if (p.func && cuFuncGetName(&name, p.func) == CUDA_SUCCESS && name) return name;
    if (p.kern && cuKernelGetName(&name, p.kern) == CUDA_SUCCESS && name) return name;
    return "";
}

const char* node_type_name(CUgraphNodeType t) {
    switch (t) {
        case CU_GRAPH_NODE_TYPE_KERNEL: return "KERNEL";
        case CU_GRAPH_NODE_TYPE_MEMCPY: return "MEMCPY";
        case CU_GRAPH_NODE_TYPE_MEMSET: return "MEMSET";
        case CU_GRAPH_NODE_TYPE_HOST: return "HOST";
        case CU_GRAPH_NODE_TYPE_GRAPH: return "CHILD_GRAPH";
        case CU_GRAPH_NODE_TYPE_EMPTY: return "EMPTY";
        case CU_GRAPH_NODE_TYPE_WAIT_EVENT: return "EVENT_WAIT";
        case CU_GRAPH_NODE_TYPE_EVENT_RECORD: return "EVENT_RECORD";
        case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL: return "EXT_SEM_SIGNAL";
        case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: return "EXT_SEM_WAIT";
        case CU_GRAPH_NODE_TYPE_MEM_ALLOC: return "MEM_ALLOC";
        case CU_GRAPH_NODE_TYPE_MEM_FREE: return "MEM_FREE";
        case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: return "BATCH_MEM_OP";
        case CU_GRAPH_NODE_TYPE_CONDITIONAL: return "CONDITIONAL";
        default: return "OTHER";
    }
}

// Writes {"label", "n_nodes", "n_edges", "nodes":[...], "edges":[[a,b],...]}.
// Node ids are the positions in cuGraphGetNodes order. Returns false on any driver error.
bool dump_graph_stream(CUgraph g, std::ostream& os, const std::string& label) {
    size_t n = 0;
    if (cuGraphGetNodes(g, nullptr, &n) != CUDA_SUCCESS) return false;
    std::vector<CUgraphNode> nodes(n);
    if (n && cuGraphGetNodes(g, nodes.data(), &n) != CUDA_SUCCESS) return false;
    std::unordered_map<CUgraphNode, size_t> id;
    for (size_t i = 0; i < n; ++i) id[nodes[i]] = i;
    os << "{\"label\":\"" << jesc(label) << "\",\"n_nodes\":" << n << ",\"nodes\":[";
    for (size_t i = 0; i < n; ++i) {
        CUgraphNodeType t{};
        cuGraphNodeGetType(nodes[i], &t);
        os << (i ? ",\n" : "\n") << "{\"id\":" << i << ",\"type\":\"" << node_type_name(t) << "\"";
        if (t == CU_GRAPH_NODE_TYPE_KERNEL) {
            CUDA_KERNEL_NODE_PARAMS p;
            std::memset(&p, 0, sizeof(p));
            if (cuGraphKernelNodeGetParams(nodes[i], &p) == CUDA_SUCCESS) {
                os << ",\"symbol\":\"" << jesc(kernel_symbol(p)) << "\",\"grid\":[" << p.gridDimX
                   << "," << p.gridDimY << "," << p.gridDimZ << "],\"block\":[" << p.blockDimX
                   << "," << p.blockDimY << "," << p.blockDimZ << "],\"smem\":" << p.sharedMemBytes;
                try {
                    KInfo k = read_kernel(nodes[i], static_cast<int>(i));
                    os << ",\"param_bytes\":" << k.param_bytes << ",\"param_hash\":\""
                       << hex64(fnv1a(k.blob.data(), k.param_bytes)) << "\",\"param_head\":\""
                       << hex_bytes(k.blob.data(), std::min<size_t>(k.param_bytes, 48))
                       << "\",\"packed_extra\":" << (k.use_extra ? "true" : "false");
                } catch (const std::exception& e) {
                    os << ",\"param_error\":\"" << jesc(e.what()) << "\"";
                }
            }
        }
        os << "}";
    }
    size_t ne = 0;
    if (cuGraphGetEdges_v2(g, nullptr, nullptr, nullptr, &ne) != CUDA_SUCCESS) return false;
    std::vector<CUgraphNode> ef(ne), et(ne);
    std::vector<CUgraphEdgeData> ed(ne);
    if (ne && cuGraphGetEdges_v2(g, ef.data(), et.data(), ed.data(), &ne) != CUDA_SUCCESS) return false;
    os << "\n],\"n_edges\":" << ne << ",\"edges\":[";
    for (size_t i = 0; i < ne; ++i) {
        os << (i ? "," : "") << "[" << id[ef[i]] << "," << id[et[i]] << "," << static_cast<int>(ed[i].type)
           << "]";
    }
    os << "]}\n";
    return true;
}

bool dump_graph_file(CUgraph g, const std::string& path, const std::string& label) {
    std::ofstream f(path);
    if (!f) return false;
    return dump_graph_stream(g, f, label);
}

size_t used_bytes() {
    size_t f = 0, t = 0;
    cuMemGetInfo(&f, &t);
    return t - f;
}

// Makes the context that owns `stream` current for the duration of the fold. Several driver
// calls (attribute copy, graph node creation) consult the current context. A stream created
// by cuGreenCtxStreamCreate belongs to a green context; cuStreamGetCtx_v2 returns it, and
// cuCtxFromGreenCtx turns it into a pushable CUcontext.
struct CtxScope {
    bool pushed = false;
    int mode = 0;  // 0 nothing pushed, 1 pushed stream's context, 2 pushed node context, 3 failed
    CtxScope(CUstream stream, CUcontext node_ctx) {
        CUcontext cur = nullptr;
        cuCtxGetCurrent(&cur);
        CUcontext cand[2] = {nullptr, node_ctx};
        if (stream) {
            CUcontext c = nullptr;
            CUgreenCtx g = nullptr;
            if (cuStreamGetCtx_v2(stream, &c, &g) == CUDA_SUCCESS) {
                if (g) {
                    CUcontext from_green = nullptr;
                    if (cuCtxFromGreenCtx(&from_green, g) == CUDA_SUCCESS) c = from_green;
                }
                cand[0] = c;
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (!cand[i]) continue;
            if (cand[i] == cur) { mode = 0; return; }
            if (cuCtxPushCurrent(cand[i]) == CUDA_SUCCESS) {
                pushed = true;
                mode = i + 1;
                return;
            }
        }
        mode = 3;
    }
    ~CtxScope() {
        if (pushed) cuCtxPopCurrent(nullptr);
    }
};

}  // namespace

struct GfHandle {
    CUgraph graph = nullptr;
    CUgraphExec exec = nullptr;
    CUcontext ctx = nullptr;
    int* d_state = nullptr;  // idx, status, trips
    GfUpd* d_upd = nullptr;
    unsigned char* d_table = nullptr;
    // Parameter storage the scheduler/reset nodes were created from.
    cudaGraphConditionalHandle cond{};
    GfUpd* p_upd = nullptr;
    int p_n_upd = 0;
    unsigned char* p_table = nullptr;
    unsigned long long p_row = 0;
    int p_iters = 0;
    int* p_idx = nullptr;
    int* p_status = nullptr;
    int* p_trips = nullptr;
};

extern "C" int gf_dump_graph(void* cugraph, const char* path, const char* label) {
    return dump_graph_file(reinterpret_cast<CUgraph>(cugraph), path, label ? label : "") ? 0 : 1;
}

extern "C" int gf_fold(void* graph_ptr, void* stream, int min_iters, int min_saved_nodes, int dry_run,
                       const char* dump_dir, GfReport* rep, GfHandle** out_handle) {
    std::memset(rep, 0, sizeof(*rep));
    if (out_handle) *out_handle = nullptr;
    GfHandle* h = nullptr;
    const std::string dump = (dump_dir && dump_dir[0]) ? std::string(dump_dir) + "/" : std::string();
    try {
        CUgraph orig = reinterpret_cast<CUgraph>(graph_ptr);
        if (!dump.empty()) dump_graph_file(orig, dump + "graph_before.json", "before fold");
        size_t n_nodes = 0;
        DRV(cuGraphGetNodes(orig, nullptr, &n_nodes));
        std::vector<CUgraphNode> nodes(n_nodes);
        DRV(cuGraphGetNodes(orig, nodes.data(), &n_nodes));
        std::unordered_map<CUgraphNode, int> gidx_of;
        for (size_t i = 0; i < n_nodes; ++i) gidx_of[nodes[i]] = static_cast<int>(i);

        std::vector<CUgraphNodeType> types(n_nodes);
        std::vector<int> kernels;  // gidx of kernel nodes in creation order
        for (size_t i = 0; i < n_nodes; ++i) {
            DRV(cuGraphNodeGetType(nodes[i], &types[i]));
            if (types[i] == CU_GRAPH_NODE_TYPE_KERNEL) kernels.push_back(static_cast<int>(i));
            if (types[i] == CU_GRAPH_NODE_TYPE_EVENT_RECORD) rep->n_event_record++;
            if (types[i] == CU_GRAPH_NODE_TYPE_WAIT_EVENT) rep->n_event_wait++;
        }
        rep->n_nodes = static_cast<int64_t>(n_nodes);
        rep->n_kernels = static_cast<int64_t>(kernels.size());
        if (kernels.size() < 8) throw Fail("too few kernel nodes");

        // edges
        size_t n_edges = 0;
        DRV(cuGraphGetEdges_v2(orig, nullptr, nullptr, nullptr, &n_edges));
        std::vector<CUgraphNode> efrom(n_edges), eto(n_edges);
        std::vector<CUgraphEdgeData> edata(n_edges);
        if (n_edges) DRV(cuGraphGetEdges_v2(orig, efrom.data(), eto.data(), edata.data(), &n_edges));
        for (size_t i = 0; i < n_edges; ++i) {
            if (edata[i].type != CU_GRAPH_DEPENDENCY_TYPE_DEFAULT || edata[i].from_port != 0)
                rep->n_nondefault_edges++;
        }
        rep->n_edges = static_cast<int64_t>(n_edges);

        // 1. kernel info and launch classes
        std::vector<KInfo> ks;
        ks.reserve(kernels.size());
        std::vector<int> reps_idx;  // representative per class
        std::vector<int> cls(kernels.size());
        for (size_t i = 0; i < kernels.size(); ++i) {
            ks.push_back(read_kernel(nodes[kernels[i]], kernels[i]));
            int found = -1;
            for (int c = 0; c < static_cast<int>(reps_idx.size()); ++c) {
                if (same_launch(ks[reps_idx[c]], ks.back())) {
                    found = c;
                    break;
                }
            }
            if (found < 0) {
                found = static_cast<int>(reps_idx.size());
                reps_idx.push_back(static_cast<int>(i));
            }
            ks.back().cls = found;
            cls[i] = found;
        }
        rep->n_classes = static_cast<int64_t>(reps_idx.size());

        // 2. periodic region
        std::vector<PeriodCandidate> period_trace;
        Region reg = find_region(cls, std::max(min_iters, 3), dump.empty() ? nullptr : &period_trace);
        if (!dump.empty()) {
            // Written before the region check so a refused fold still leaves its analysis behind.
            std::ofstream f(dump + "fold_classes.json");
            f << "{\"n_kernels\":" << kernels.size() << ",\"n_classes\":" << reps_idx.size()
              << ",\"classes\":[";
            std::vector<int> count(reps_idx.size(), 0);
            for (int c : cls) count[c]++;
            for (size_t c = 0; c < reps_idx.size(); ++c) {
                const KInfo& k = ks[reps_idx[c]];
                f << (c ? ",\n" : "\n") << "{\"id\":" << c << ",\"count\":" << count[c]
                  << ",\"symbol\":\"" << jesc(kernel_symbol(k.p)) << "\",\"grid\":[" << k.p.gridDimX << ","
                  << k.p.gridDimY << "," << k.p.gridDimZ << "],\"block\":[" << k.p.blockDimX << ","
                  << k.p.blockDimY << "," << k.p.blockDimZ << "],\"smem\":" << k.p.sharedMemBytes
                  << ",\"param_bytes\":" << k.param_bytes << "}";
            }
            // run-length encoding of the class sequence in node creation order
            f << "\n],\"sequence_rle\":[";
            for (size_t i = 0, first = 1; i < cls.size();) {
                size_t j = i;
                while (j < cls.size() && cls[j] == cls[i]) ++j;
                f << (first ? "" : ",") << "[" << cls[i] << "," << (j - i) << "]";
                first = 0;
                i = j;
            }
            f << "],\"period_candidates\":[";
            for (size_t i = 0; i < period_trace.size(); ++i) {
                const auto& c = period_trace[i];
                f << (i ? "," : "") << "{\"P\":" << c.P << ",\"start\":" << c.S << ",\"iters\":" << c.iters
                  << ",\"cover\":" << c.cover << "}";
            }
            f << "]}\n";
        }
        if (reg.iters < std::max(min_iters, 3)) throw Fail("no periodic kernel region");
        rep->region_start = reg.S;
        rep->period = reg.P;
        rep->iters = reg.iters;
        const int K0 = reg.S;
        const int K1 = reg.S + reg.iters * reg.P;  // exclusive
        rep->prefix_kernels = K0;
        rep->suffix_kernels = static_cast<int64_t>(kernels.size()) - K1;
        const int g_lo = ks[K0].gidx;
        const int g_hi = ks[K1 - 1].gidx;

        // loop membership by gidx
        std::vector<int> loop_iter(n_nodes, -1), loop_pos(n_nodes, -1);
        std::vector<char> in_loop_kernel(n_nodes, 0), in_loop_event(n_nodes, 0);
        for (int k = K0; k < K1; ++k) {
            in_loop_kernel[ks[k].gidx] = 1;
            loop_iter[ks[k].gidx] = (k - K0) / reg.P;
            loop_pos[ks[k].gidx] = (k - K0) % reg.P;
        }
        for (int g = g_lo; g <= g_hi; ++g) {
            if (in_loop_kernel[g]) continue;
            if (types[g] == CU_GRAPH_NODE_TYPE_EVENT_RECORD || types[g] == CU_GRAPH_NODE_TYPE_EMPTY) {
                in_loop_event[g] = 1;
                rep->loop_events++;
            } else {
                throw Fail("loop region contains a node type the WHILE body cannot take");
            }
        }

        // adjacency with the loop's EVENT_RECORD / EMPTY nodes bypassed
        std::vector<std::set<int>> succ(n_nodes), pred(n_nodes);
        std::set<std::pair<int, int>> orig_edges;
        for (size_t i = 0; i < n_edges; ++i) {
            int a = gidx_of.at(efrom[i]), b = gidx_of.at(eto[i]);
            succ[a].insert(b);
            pred[b].insert(a);
            orig_edges.insert({a, b});
        }
        for (int g = g_lo; g <= g_hi; ++g) {
            if (!in_loop_event[g]) continue;
            for (int p : pred[g]) {
                succ[p].erase(g);
                for (int s : succ[g])
                    if (s != p) succ[p].insert(s);
            }
            for (int s : succ[g]) {
                pred[s].erase(g);
                for (int p : pred[g])
                    if (s != p) pred[s].insert(p);
            }
            succ[g].clear();
            pred[g].clear();
        }

        std::set<int> in_deps, out_succ;
        std::set<std::pair<int, int>> body_edges;       // (pos, pos), union over iterations
        std::vector<std::pair<int, int>> added_edges;   // retained->retained created by bypass
        for (int a = 0; a < static_cast<int>(n_nodes); ++a) {
            if (in_loop_event[a]) continue;
            for (int b : succ[a]) {
                bool la = in_loop_kernel[a], lb = in_loop_kernel[b];
                if (!la && !lb) {
                    if (!orig_edges.count({a, b})) added_edges.emplace_back(a, b);
                } else if (!la && lb) {
                    in_deps.insert(a);
                } else if (la && !lb) {
                    out_succ.insert(b);
                } else {
                    if (loop_iter[a] > loop_iter[b]) throw Fail("edge goes to an earlier iteration");
                    if (loop_iter[a] == loop_iter[b]) body_edges.insert({loop_pos[a], loop_pos[b]});
                    else rep->cross_edges++;
                }
            }
        }
        rep->in_edges = static_cast<int64_t>(in_deps.size());
        rep->out_edges = static_cast<int64_t>(out_succ.size());
        rep->body_edges = static_cast<int64_t>(body_edges.size());

        // 3. parameter spans that differ between iterations
        struct Span {
            int pos;
            uint32_t off, size;
            uint64_t table_off;
        };
        std::vector<Span> spans;
        uint64_t row_bytes = 0;
        for (int j = 0; j < reg.P; ++j) {
            const KInfo& k0 = ks[K0 + j];
            const size_t words = (k0.param_bytes + 7) / 8;
            std::vector<char> diff(words, 0);
            for (int it = 1; it < reg.iters; ++it) {
                const KInfo& kt = ks[K0 + it * reg.P + j];
                if (kt.param_bytes != k0.param_bytes) throw Fail("parameter buffer size changed");
                for (size_t w = 0; w < words; ++w) {
                    size_t o = w * 8, n = std::min<size_t>(8, k0.param_bytes - o);
                    if (std::memcmp(k0.blob.data() + o, kt.blob.data() + o, n) != 0) diff[w] = 1;
                }
            }
            size_t w = 0;
            while (w < words) {
                if (!diff[w]) { ++w; continue; }
                size_t e = w;
                // merge differing words separated by at most one clean word
                while (e + 1 < words && (diff[e + 1] || (e + 2 < words && diff[e + 2]))) ++e;
                size_t off = w * 8;
                size_t end = std::min(k0.param_bytes, (e + 1) * 8);
                Span s{j, static_cast<uint32_t>(off), static_cast<uint32_t>(end - off), row_bytes};
                spans.push_back(s);
                row_bytes += (end - off + 15) & ~size_t(15);
                w = e + 1;
            }
        }
        rep->n_updates = static_cast<int64_t>(spans.size());
        rep->row_bytes = static_cast<int64_t>(row_bytes);
        rep->table_bytes = static_cast<int64_t>(row_bytes * reg.iters);
        rep->unrolled_nodes = rep->n_nodes;
        // predicted parent-graph nodes: region removed, WHILE and reset added
        // (the body of reg.P + 1 nodes is counted separately in body_nodes)
        rep->folded_nodes = rep->n_nodes - static_cast<int64_t>(reg.iters * reg.P) -
                            rep->loop_events + 2;
        if (!dump.empty()) {
            std::ofstream f(dump + "fold_analysis.json");
            f << "{\"region\":{\"start_kernel\":" << K0 << ",\"period\":" << reg.P << ",\"iters\":" << reg.iters
              << ",\"end_kernel_exclusive\":" << K1 << ",\"first_gidx\":" << g_lo << ",\"last_gidx\":" << g_hi
              << ",\"prefix_kernels\":" << rep->prefix_kernels << ",\"suffix_kernels\":" << rep->suffix_kernels
              << ",\"loop_events_removed\":" << rep->loop_events << "},\n";
            f << "\"body_positions\":[";
            for (int j = 0; j < reg.P; ++j) {
                const KInfo& k = ks[K0 + j];
                f << (j ? ",\n" : "\n") << "{\"pos\":" << j << ",\"class\":" << k.cls << ",\"gidx_iter0\":"
                  << k.gidx << ",\"symbol\":\"" << jesc(kernel_symbol(k.p)) << "\",\"grid\":[" << k.p.gridDimX
                  << "," << k.p.gridDimY << "," << k.p.gridDimZ << "],\"block\":[" << k.p.blockDimX << ","
                  << k.p.blockDimY << "," << k.p.blockDimZ << "],\"param_bytes\":" << k.param_bytes << "}";
            }
            f << "\n],\n\"in_deps_gidx\":[";
            { bool first = true; for (int a : in_deps) { f << (first ? "" : ",") << a; first = false; } }
            f << "],\"out_succ_gidx\":[";
            { bool first = true; for (int b : out_succ) { f << (first ? "" : ",") << b; first = false; } }
            f << "],\"body_edges\":[";
            { bool first = true; for (auto& e : body_edges) { f << (first ? "" : ",") << "[" << e.first << "," << e.second << "]"; first = false; } }
            f << "],\"cross_iteration_edges\":" << rep->cross_edges << ",\"bypass_edges_added\":" << added_edges.size()
              << ",\n\"updates\":[";
            for (size_t u = 0; u < spans.size(); ++u) {
                const auto& s = spans[u];
                f << (u ? ",\n" : "\n") << "{\"pos\":" << s.pos << ",\"offset\":" << s.off << ",\"size\":" << s.size
                  << ",\"table_off\":" << s.table_off << ",\"values\":[";
                for (int it = 0; it < reg.iters; ++it) {
                    const KInfo& kt = ks[K0 + it * reg.P + s.pos];
                    f << (it ? "," : "") << "\"" << hex_bytes(kt.blob.data() + s.off, std::min<uint32_t>(s.size, 32)) << "\"";
                }
                f << "]}";
            }
            f << "\n],\"table\":{\"row_bytes\":" << row_bytes << ",\"iters\":" << reg.iters
              << ",\"total_bytes\":" << row_bytes * reg.iters << "},\"dry_run\":" << (dry_run ? "true" : "false")
              << "}\n";
        }
        {
            // Nodes after the fold: parent graph + WHILE body. A small saving does not pay for the
            // scheduler kernel the body runs per iteration (executable graph memory is ~4 KiB/node).
            const int64_t net_nodes = rep->folded_nodes + reg.P + 1;
            const int64_t saved = rep->n_nodes - net_nodes;
            if (saved < min_saved_nodes) {
                throw Fail("fold not worthwhile: the region saves " + std::to_string(saved) +
                           " nodes, below the minimum of " + std::to_string(min_saved_nodes));
            }
        }
        if (dry_run) {
            std::snprintf(rep->message, sizeof(rep->message), "dry run ok");
            return 0;
        }

        // 4. build the folded graph
        // The node context of the captured kernels (a Green Context stays a Green Context).
        // It is used as is: no cuCtxPushCurrent, since the context recorded on nodes of
        // cuGreenCtxStreamCreate streams is not a pushable CUcontext.
        CUcontext ctx = ks[K0].p.ctx;
        if (!ctx) cuCtxGetCurrent(&ctx);
        CtxScope scope(reinterpret_cast<CUstream>(stream), ctx);
        rep->ctx_mode = scope.mode;
        h = new GfHandle();
        h->ctx = ctx;
        size_t mem0 = used_bytes();

        CUgraph clone = nullptr;
        DRV(cuGraphClone(&clone, orig));
        h->graph = clone;
        auto clone_of = [&](int g) {
            CUgraphNode c = nullptr;
            DRV(cuGraphNodeFindInClone(&c, nodes[g], clone));
            return c;
        };
        // retained->retained edges that only existed through a bypassed event
        std::vector<CUgraphNode> add_from, add_to;
        for (auto& e : added_edges) {
            add_from.push_back(clone_of(e.first));
            add_to.push_back(clone_of(e.second));
        }
        std::vector<CUgraphNode> dep_clones, succ_clones;
        for (int a : in_deps) dep_clones.push_back(clone_of(a));
        for (int b : out_succ) succ_clones.push_back(clone_of(b));
        // delete the region from the clone
        for (int g = g_lo; g <= g_hi; ++g) {
            if (in_loop_kernel[g] || in_loop_event[g]) DRV(cuGraphDestroyNode(clone_of(g)));
        }
        if (!add_from.empty())
            DRV(cuGraphAddDependencies_v2(clone, add_from.data(), add_to.data(), nullptr, add_from.size()));

        // device state
        DRV(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&h->d_state), 3 * sizeof(int)));
        DRV(cuMemsetD8(reinterpret_cast<CUdeviceptr>(h->d_state), 0, 3 * sizeof(int)));
        h->p_idx = h->d_state;
        h->p_status = h->d_state + 1;
        h->p_trips = h->d_state + 2;

        // WHILE node
        CUgraphConditionalHandle drv_cond = 0;  // same 64-bit opaque value as the runtime type
        DRV(cuGraphConditionalHandleCreate(&drv_cond, clone, ctx, 1, CU_GRAPH_COND_ASSIGN_DEFAULT));
        h->cond = static_cast<cudaGraphConditionalHandle>(drv_cond);
        // reset kernel (root of the folded graph, before the WHILE)
        // Context-independent kernel handles: valid for whichever context the node names.
        cudaKernel_t f_reset = nullptr, f_sched = nullptr;
        RT(cudaGetKernel(&f_reset, reinterpret_cast<const void*>(gf_reset)));
        RT(cudaGetKernel(&f_sched, reinterpret_cast<const void*>(gf_sched)));
        void* reset_args[3] = {&h->p_idx, &h->p_status, &h->p_trips};
        CUDA_KERNEL_NODE_PARAMS rp;
        std::memset(&rp, 0, sizeof(rp));
        rp.kern = reinterpret_cast<CUkernel>(f_reset);
        rp.gridDimX = rp.gridDimY = rp.gridDimZ = 1;
        rp.blockDimX = rp.blockDimY = rp.blockDimZ = 1;
        rp.kernelParams = reset_args;
        rp.ctx = ctx;
        CUgraphNode reset_node = nullptr;
        DRV(cuGraphAddKernelNode(&reset_node, clone, nullptr, 0, &rp));

        CUgraphNodeParams cp;
        std::memset(&cp, 0, sizeof(cp));
        cp.type = CU_GRAPH_NODE_TYPE_CONDITIONAL;
        cp.conditional.handle = drv_cond;
        cp.conditional.type = CU_GRAPH_COND_TYPE_WHILE;
        cp.conditional.size = 1;
        cp.conditional.ctx = ctx;
        std::vector<CUgraphNode> while_deps = dep_clones;
        while_deps.push_back(reset_node);
        CUgraphNode while_node = nullptr;
        DRV(cuGraphAddNode(&while_node, clone, while_deps.data(), while_deps.size(), &cp));
        CUgraph body = cp.conditional.phGraph_out[0];
        if (!body) throw Fail("WHILE node returned no body graph");
        if (!succ_clones.empty()) {
            std::vector<CUgraphNode> from(succ_clones.size(), while_node);
            DRV(cuGraphAddDependencies_v2(clone, from.data(), succ_clones.data(), nullptr, from.size()));
        }

        // body: iteration 0
        std::vector<std::vector<void*>> kptrs(reg.P);
        std::vector<std::array<void*, 5>> extra_arr(reg.P);
        std::vector<CUgraphNode> bnode(reg.P);
        std::vector<char> needs_dev(reg.P, 0);
        for (auto& s : spans) needs_dev[s.pos] = 1;
        std::vector<cudaGraphDeviceNode_t> devnode(reg.P, nullptr);
        for (int j = 0; j < reg.P; ++j) {
            KInfo& k = ks[K0 + j];
            CUDA_KERNEL_NODE_PARAMS np = k.p;
            if (k.use_extra) {
                extra_arr[j] = {CU_LAUNCH_PARAM_BUFFER_POINTER, k.blob.data(), CU_LAUNCH_PARAM_BUFFER_SIZE,
                                &k.param_bytes, CU_LAUNCH_PARAM_END};
                np.kernelParams = nullptr;
                np.extra = extra_arr[j].data();
            } else {
                kptrs[j].resize(k.pinfo.size());
                for (size_t i = 0; i < k.pinfo.size(); ++i) kptrs[j][i] = k.blob.data() + k.pinfo[i].first;
                np.kernelParams = kptrs[j].data();
                np.extra = nullptr;
            }
            DRV(cuGraphAddKernelNode(&bnode[j], body, nullptr, 0, &np));
            // Launch attributes one by one (cuGraphKernelNodeCopyAttributes rejects whole
            // sets that contain attributes a graph node cannot take).
            //  - PROGRAMMATIC_STREAM_SERIALIZATION (PDL): dropped. The folded graph orders
            //    kernels with full dependencies, which is stronger; PDL-aware kernels run
            //    correctly without the early-launch overlap.
            //  - PROGRAMMATIC_EVENT / LAUNCH_COMPLETION_EVENT: refused when set.
            //  - others: copied; a failure on a non-default value refuses the fold.
            for (int id = 1; id <= kMaxAttrId; ++id) {
                CUkernelNodeAttrValue v;
                std::memset(&v, 0, sizeof(v));
                if (cuGraphKernelNodeGetAttribute(k.node, static_cast<CUkernelNodeAttrID>(id), &v) != CUDA_SUCCESS)
                    continue;
                const unsigned char* vb = reinterpret_cast<const unsigned char*>(&v);
                bool nonzero = false;
                for (size_t b = 0; b < sizeof(v); ++b) nonzero |= vb[b] != 0;
                if (id == CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION) {
                    if (nonzero) rep->pdl_attr_dropped++;
                    continue;
                }
                if (!nonzero) continue;  // default value
                CUresult rs = cuGraphKernelNodeSetAttribute(bnode[j], static_cast<CUkernelNodeAttrID>(id), &v);
                if (rs != CUDA_SUCCESS) {
                    unsigned long long head = 0;
                    std::memcpy(&head, vb, sizeof(head));
                    throw Fail("cannot copy launch attribute id " + std::to_string(id) + " (rc " +
                               std::to_string(static_cast<int>(rs)) + ", value head 0x" +
                               [&] { char t[32]; std::snprintf(t, sizeof(t), "%llx", head); return std::string(t); }() +
                               ") of body kernel " + std::to_string(j));
                }
            }
            if (needs_dev[j]) {
                CUkernelNodeAttrValue v;
                std::memset(&v, 0, sizeof(v));
                v.deviceUpdatableKernelNode.deviceUpdatable = 1;
                DRV(cuGraphKernelNodeSetAttribute(bnode[j], CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE, &v));
                devnode[j] = reinterpret_cast<cudaGraphDeviceNode_t>(v.deviceUpdatableKernelNode.devNode);
                if (!devnode[j]) throw Fail("device-updatable attribute returned a null node");
            }
        }
        std::vector<char> has_succ(reg.P, 0);
        {
            std::vector<CUgraphNode> bf, bt;
            for (auto& e : body_edges) {
                bf.push_back(bnode[e.first]);
                bt.push_back(bnode[e.second]);
                has_succ[e.first] = 1;
            }
            if (!bf.empty()) DRV(cuGraphAddDependencies_v2(body, bf.data(), bt.data(), nullptr, bf.size()));
        }

        // table + update list
        std::vector<GfUpd> upd(spans.size());
        std::vector<unsigned char> table(row_bytes * reg.iters, 0);
        for (size_t u = 0; u < spans.size(); ++u) {
            upd[u].node = devnode[spans[u].pos];
            upd[u].offset = spans[u].off;
            upd[u].size = spans[u].size;
            upd[u].table_off = spans[u].table_off;
            for (int it = 0; it < reg.iters; ++it) {
                const KInfo& kt = ks[K0 + it * reg.P + spans[u].pos];
                std::memcpy(table.data() + it * row_bytes + spans[u].table_off,
                            kt.blob.data() + spans[u].off, spans[u].size);
            }
        }
        if (!upd.empty()) {
            DRV(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&h->d_upd), upd.size() * sizeof(GfUpd)));
            DRV(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(h->d_upd), upd.data(), upd.size() * sizeof(GfUpd)));
            DRV(cuMemAlloc(reinterpret_cast<CUdeviceptr*>(&h->d_table), table.size()));
            DRV(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(h->d_table), table.data(), table.size()));
        }
        h->p_upd = h->d_upd;
        h->p_n_upd = static_cast<int>(upd.size());
        h->p_table = h->d_table;
        h->p_row = row_bytes;
        h->p_iters = reg.iters;

        void* sched_args[9] = {&h->cond, &h->p_upd, &h->p_n_upd, &h->p_table, &h->p_row,
                               &h->p_iters, &h->p_idx, &h->p_status, &h->p_trips};
        CUDA_KERNEL_NODE_PARAMS sp;
        std::memset(&sp, 0, sizeof(sp));
        sp.kern = reinterpret_cast<CUkernel>(f_sched);
        sp.gridDimX = sp.gridDimY = sp.gridDimZ = 1;
        sp.blockDimX = 128;
        sp.blockDimY = sp.blockDimZ = 1;
        sp.kernelParams = sched_args;
        sp.ctx = ctx;
        std::vector<CUgraphNode> sinks;
        for (int j = 0; j < reg.P; ++j)
            if (!has_succ[j]) sinks.push_back(bnode[j]);
        CUgraphNode sched = nullptr;
        DRV(cuGraphAddKernelNode(&sched, body, sinks.data(), sinks.size(), &sp));

        if (!dump.empty()) {
            dump_graph_file(clone, dump + "graph_after_parent.json", "after fold: parent graph (WHILE node is CONDITIONAL)");
            dump_graph_file(body, dump + "graph_after_body.json", "after fold: WHILE body = iteration 0 + gf_sched");
        }
        DRV(cuGraphInstantiateWithFlags(&h->exec, clone, 0));
        size_t mem1 = used_bytes();
        rep->instantiate_driver_delta = static_cast<int64_t>(mem1) - static_cast<int64_t>(mem0);
        size_t cn = 0;
        DRV(cuGraphGetNodes(clone, nullptr, &cn));
        rep->folded_nodes_actual = static_cast<int64_t>(cn);
        rep->body_nodes = reg.P + 1;
        std::snprintf(rep->message, sizeof(rep->message),
                      "folded %lld nodes -> %lld parent + %d body; period %d x %d iterations; "
                      "%lld updates, %lld B/iter table",
                      static_cast<long long>(rep->n_nodes), static_cast<long long>(cn), reg.P + 1,
                      reg.P, reg.iters, static_cast<long long>(rep->n_updates),
                      static_cast<long long>(rep->row_bytes));
        *out_handle = h;
        return 0;
    } catch (const std::exception& e) {
        rep->status = 1;
        std::snprintf(rep->message, sizeof(rep->message), "%s", e.what());
        if (h) {
            if (h->exec) cuGraphExecDestroy(h->exec);
            if (h->graph) cuGraphDestroy(h->graph);
            if (h->d_state) cuMemFree(reinterpret_cast<CUdeviceptr>(h->d_state));
            if (h->d_upd) cuMemFree(reinterpret_cast<CUdeviceptr>(h->d_upd));
            if (h->d_table) cuMemFree(reinterpret_cast<CUdeviceptr>(h->d_table));
            delete h;
        }
        return 1;
    }
}

extern "C" int gf_launch(GfHandle* h, void* stream) {
    return static_cast<int>(cuGraphLaunch(h->exec, reinterpret_cast<CUstream>(stream)));
}

// Debug: iteration counter, scheduler error code, trips of the last launch.
extern "C" int gf_state(GfHandle* h, int64_t* out3) {
    int v[3] = {0, 0, 0};
    CUresult r = cuMemcpyDtoH(v, reinterpret_cast<CUdeviceptr>(h->d_state), sizeof(v));
    for (int i = 0; i < 3; ++i) out3[i] = v[i];
    return static_cast<int>(r);
}

extern "C" void gf_destroy(GfHandle* h) {
    if (!h) return;
    if (h->exec) cuGraphExecDestroy(h->exec);
    if (h->graph) cuGraphDestroy(h->graph);
    if (h->d_state) cuMemFree(reinterpret_cast<CUdeviceptr>(h->d_state));
    if (h->d_upd) cuMemFree(reinterpret_cast<CUdeviceptr>(h->d_upd));
    if (h->d_table) cuMemFree(reinterpret_cast<CUdeviceptr>(h->d_table));
    delete h;
}
