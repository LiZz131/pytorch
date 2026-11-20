# cuda_graph_memopt：CUDA Graph 显存优化工具（context 文档）

本文档说明 `cuda_graph_memopt` 的动机、安装、使用、原理、算法、正确性保证和全部改动。本工具已合并进 `sglang_prefill_miss`（与 share-prefix overlap 的 `oar_done` 方案同一棵树）。工具自身的测试结果在 `bench_out/cuda_graph_memopt/TEST_RESULTS.md`，合并后的验收结果在 `bench_out/cgmo_merge/MERGE_RESULTS.md`。

目录约定（相对仓库根 `sglang_prefill_miss/`）：

```
cuda_graph_memopt/            工具本体（纯 Python + 两个原生库）
  config.py                   环境变量 -> Config
  manager.py                  GraphMemOpt：capture / replay / verify / summary
  tensor/plan.py              静态张量摆放（贪心按大小）与不变量检查
  tensor/runtime.py           记录 -> 规划 -> 打包捕获（ctypes 调用打过补丁的 libc10_cuda.so）
  fold/graph_fold.cu|.h       层折叠（把重复层折成 WHILE 条件节点）
  fold/lib.py                 graph_fold.so 的 ctypes 前端、按需编译
  debug/dump.py               debug 模式的全部 dump
  build.sh                    编译 libc10_cuda.so 与 graph_fold.so
  patches/pytorch-c10-tmo.patch  对 PyTorch c10/cuda/CUDACachingAllocator.cpp 的补丁（同一份补丁已提交到 PyTorch fork）
  tests/                      CPU / GPU 单元测试与 Green Context 测试（`CGMO_TEST_OUT` 指定输出目录）
python/sglang/srt/model_executor/cgmo_hook.py   SGLang 接入胶水
```

## 1. 动机

DeepSeek-V3.1 + pdmux（每个 batch size、每个 Green Context stream 组各捕获一张解码图，默认 20 个 bs × 8 组 = 160 张）下，CUDA Graph 在每张 GPU 上有两笔显存开销：

1. **私有 pool（PyTorch caching allocator）**：捕获期间的中间张量由 allocator 在线 first-fit 切分。一块刚释放的大块被小请求切开后，下一次同样大小的请求放不回原处，只能再 `cudaMalloc` 一段。地址已烤进 graph，`empty_cache` 不会归还。实测最大 bs 的一次捕获，活跃峰值 1.87 GiB，pool 却保留 3.71 GiB。
2. **可执行体 `cudaGraphExec_t`**：由驱动分配，不在 PyTorch allocator 里。每张图约 2073 个节点、约 40 MiB，160 张共约 6.25 GiB。层与层的 kernel 序列几乎相同（周期 32 个 kernel，重复 58 次）。

工具用两个互相独立的 feature 解决这两笔开销，且都保持数值逐位一致、与 Green Context（pdmux 的 SM 分区）兼容：

| feature | 对象 | 做法 | 数据（DeepSeek-V3.1，每 GPU） |
| --- | --- | --- | --- |
| `tensor` | 私有 pool | 捕获两遍：第一遍记录分配轨迹，规划偏移，第二遍按偏移从一段 arena 里切块 | pool 3706 MiB -> 1874 MiB |
| `fold` | 可执行体 | 把周期重复的层折成一个 WHILE 条件节点 | 约 40 MiB/图 -> 约 2.4 MiB/图 |

两者可单开、可同开（`CGMO=tensor,fold`）。未设置 `CGMO` 时工具完全不介入，SGLang 走原代码路径。

## 2. 安装

环境要求：CUDA >= 12.4（条件节点），驱动支持设备侧 `cudaGraphKernelNodeSetParam`；Green Context 需 CUDA >= 12.5（pdmux 本来就需要）；PyTorch 带 `CUDAGraph(keep_graph=True)` 与 `raw_cuda_graph()`（镜像里的 2.9.1+cu129 满足）；`nvcc` 与 `g++`。

```bash
cd sglang_prefill_miss
bash cuda_graph_memopt/build.sh all          # 或 c10 / fold；输出到 cuda_graph_memopt/lib/（已在 .gitignore）
```

- `build.sh c10`：只编译 PyTorch 源码里 `c10/cuda` 的 11 个源文件（源码目录由 `PYTORCH_SRC` 指定，默认依次找 `<repo>/third_party/pytorch`、`<repo>/3rdparty/pytorch`、`/workspace/.home/third_party/pytorch`；脚本会检查它已打补丁），链接镜像自带的 `libc10`，产出 `lib/libc10_cuda.so`（含 13 个 `tmo_*` 导出符号）。不覆盖 site-packages 里的那一份。
- `build.sh fold`：`nvcc -rdc=true -arch=native` 编译，再 `-dlink`，链接 `-lcudadevrt -lcuda`，产出 `lib/graph_fold.so`（设备侧 SetParam 需要设备运行时）。`graph_fold.so` 缺失或比源码旧时，第一次使用会在文件锁内自动编译。

运行时：

```bash
export LD_LIBRARY_PATH=$PWD/cuda_graph_memopt/lib:$LD_LIBRARY_PATH   # 必须在 python 启动前设置，tensor 才会生效
export PYTHONPATH=$PWD:$PWD/python:$PYTHONPATH                       # cuda_graph_memopt 与 sglang
export CGMO=tensor,fold
```

验证加载的是补丁版 allocator：启动后 `/proc/self/maps` 里的 `libc10_cuda.so` 路径应在 `cuda_graph_memopt/lib/` 下（`GraphMemOpt` 初始化时把该路径写进日志：`tensor: libc10_cuda=...`）。

补丁来源：`patches/pytorch-c10-tmo.patch`（针对 `c10/cuda/CUDACachingAllocator.cpp`，`git apply -p1` 可直接应用到 `third_party/pytorch`）。同一改动已作为一个提交推到 PyTorch fork `LiZz131/pytorch` 的 `feature/stream-mutex-groups` 分支（在 stream mutex groups 提交之上），所以直接用该分支的 `c10/cuda` 即可编译。若镜像的 PyTorch 版本与源码不一致，需把补丁重新应用到对应版本再 `build.sh c10`。

## 3. 使用

### 3.1 SGLang（环境变量即可）

```bash
CGMO=tensor,fold  bash test_scripts/perf/start_server_pdmux_special_dpa.sh   # 脚本在 CGMO 非空时自动设置 LD_LIBRARY_PATH
```

`CudaGraphRunner` 在 `__init__` 里调用 `cgmo_hook.create_graph_memopt()`。仅当 runner 的 `_capture_graph / _create_device_graph / capture_one_batch_size / replay` 都是基类版本且设备是 CUDA 时才启用；NPU、EAGLE draft 等覆写这些方法的 runner 保持原行为（日志里有 warning）。

### 3.2 独立使用

```python
from cuda_graph_memopt import GraphMemOpt            # CGMO 未设置时 from_env() 返回 None

opt = GraphMemOpt.from_env()
pool = torch.cuda.graph_pool_handle()
graph = opt.new_graph()                              # CUDAGraph(keep_graph=True)（fold 开启时）
out = opt.capture(graph, pool, stream, fn, key="0_128", group=0)   # 代替 `with torch.cuda.graph(...)`
opt.verify_pending()                                 # 在所有 graph_capture() 上下文退出之后
opt.replay("0_128", graph)                           # 代替 graph.replay()
print(opt.finish())                                  # 汇总（debug>=1 时同时写 summary.json）
```

对 `fn` 的约束：

- 返回输出张量，且不要在别处保存 fn 内部分配的张量引用（record 遍的张量必须能在打包捕获前释放）。
- 两次调用必须以相同顺序分配相同大小的张量（`tensor` 会运行 `fn` 两次；`fold` 只需一次）。
- 不满足时：分配器在打包遍里检查每次分配的大小并抛 `RuntimeError`，manager 捕获后回退到普通捕获（`CGMO_STRICT=1` 时直接抛出）。

### 3.3 配置（环境变量，完整列表见 `config.py` 文档字符串）

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `CGMO` | 空（关闭） | `tensor` / `fold` / `all`（= `tensor,fold`）/ `tensor,fold` / `stats`（不做优化，只开记录，用作对照基线） |
| `CGMO_TENSOR_SLACK` | 0.05 | 池上第一次捕获：arena 比 first-fit 在这次捕获里多占超过这个比例才放弃规划（见 5.1） |
| `CGMO_TENSOR_VMM` | 1 | arena 用 VMM 保留虚地址，规划后再映射物理页（物理页恰好等于规划跨度）。0 则 arena 为普通 `cudaMalloc` 段 |
| `CGMO_FOLD_MIN_ITERS` | 4 | 周期区间最少重复次数 |
| `CGMO_FOLD_MIN_SAVED_NODES` | 256 | 折叠少于这个节点数的图不折（约 1 MiB 可执行体，不值得为它付每次迭代的调度开销） |
| `CGMO_VERIFY` | 0 | 每个 stream 组前 N 张折叠图，捕获后用展开图 / 折叠图回放对比（逐位 + 耗时）。校验会实例化展开图，额外占显存，只用于验证运行 |
| `CGMO_STRICT` | 0 | 1：任何 feature 无法应用时抛异常，不回退 |
| `CGMO_DEBUG` | 0 | 0 关；1 日志文件 + 每次捕获一行 jsonl + summary.json；2 另外 dump 全部细节（见第 4 节） |
| `CGMO_DUMP_DIR` | `./cgmo_dump` | debug 输出目录 |
| `CGMO_DUMP_MAX` / `CGMO_DUMP_FILTER` / `CGMO_DUMP_RANKS` | 4 / 每个 group 的第一次捕获 / 0 | 级别 2 时 dump 哪些捕获、哪些 rank |
| `CGMO_LIB_DIR` | `<pkg>/lib` | 原生库目录 |

## 4. Debug / 日志模式

`CGMO_DEBUG=1`：`<DUMP_DIR>/rank<r>_pid<p>/` 下有 `cgmo.log`（完整 DEBUG 日志）、`captures.jsonl`（每次捕获一行：key、group、各 feature 的模式与数字、driver/reserved 显存增量、耗时）、`summary.json`。

`CGMO_DEBUG=2`：对被选中的捕获，目录 `capture_m<manager>_<seq>_<key>/` 下：

```
meta.json   report.md           捕获概况、前后对比表（人读的入口）
tensor/
  trace.jsonl                   record 遍的 alloc / free_completed 事件（序号、地址、大小）
  buffers.csv                   每个 buffer：大小、生命周期 [start,end)、规划偏移、record 遍里 allocator 给它的 segment 与段内偏移
  segments_before.json          record 遍里 caching allocator 保留的 segment
  plan.json                     峰值活跃字节、规划跨度、arena、物理页、已验证的不变量
  layout.svg                    左：record 遍（生命周期 x 地址）；右：规划后。同一比例尺，直接看到空泡
  timeline.txt                  活跃字节随分配序号变化的 ASCII 曲线
fold/
  graph_before.json             原图每个节点：类型、kernel 符号（含 demangled）、grid/block/smem、参数哈希与前 48 字节、全部边
  fold_classes.json             launch class 表、按创建顺序的 class 序列（RLE）、每个周期长度 P 的最佳周期区间候选
  fold_analysis.json            选中的区间、body 位置、进入/离开循环的依赖、迭代内的边、每个参数更新区间在每次迭代的取值
  graph_after_parent.json       折叠后的父图（WHILE 是 CONDITIONAL 节点）
  graph_after_body.json         WHILE body = 第 0 次迭代 + gf_sched
  graph_before.dot / graph_after.dot / graph_after_body.dot   Graphviz（超过 400 个节点时截取首尾）
  fold_summary.md               上述 json 的可读摘要
```

dump 不影响结果：C++ 侧的 dump 只读图；Python 侧的 `verify_plan` 只在 dump 时额外运行。

## 5. 原理：用到的 feature 与 API

### 5.1 tensor：静态张量摆放

PyTorch：

- CUDA Graph 私有 pool：`torch.cuda.graph(pool=...)`，捕获期间的分配走 `DeviceCachingAllocator::malloc`，`captures_underway` 里按 capture id 识别。块在 `free_completed`（deferred free 处理完）后才可复用。
- 补丁（`CUDACachingAllocator.cpp`，13 个导出符号）：
  - `TmoTraceState`：`note_tmo_trace` 在私有 pool 过滤通过的 `alloc` 与 `free_completed` 上记录（动作、地址、大小）。导出 `tmo_trace_begin/end/count/copy_trace`。
  - `TmoPlanState`：`tmo_set_capture_plan(sizes, offsets, n, arena_bytes)`、`tmo_set_capture_plan_at(..., base)`（在已有 arena 里规划）、`tmo_capture_plan_arena_base`、`try_tmo_malloc`（在 `malloc` 入口、设备锁内；计划未激活或不在捕获 stream 上则返回 nullptr 走原路径）、`tmo_capture_plan_cursor`、`tmo_clear_capture_plan`。第 n 次分配的舍入大小必须等于 `sizes[n]`（`TORCH_CHECK`），超出计划数同样报错。
  - VMM arena：`tmo_va_only_begin/end` 期间 arena 用 `cuMemAddressReserve` 保留虚地址；`tmo_vmm_commit` 在规划完成、第一块被交出之前 `cuMemCreate/cuMemMap/cuMemSetAccess` 物理页；`tmo_vmm_physical_bytes` 返回映射的物理字节数。

CUDA 驱动 API：`cuMemAddressReserve / cuMemCreate / cuMemMap / cuMemSetAccess`（VMM）。限制：`cudaIpcGetMemHandle` 拒绝 `cuMemMap` 地址，所以 `CGMO_TENSOR_VMM=1` 时 SGLang 的 custom allreduce 必须走 `SGLANG_MEMORY_SAVER_CUDA_GRAPH=1` 路径（捕获中先拷进 `cudaMalloc` 的 IPC scratch，再 all-reduce），`custom_all_reduce.py` 在检测到 VMM arena 时自动设置该开关。

`tensor/runtime.py` 的流程（`key` 是 pool id；每个 pool 有一个进程级 arena 记录 `_ARENAS[pool] = (base, bytes)`）：

```
capture(graph, pool, stream, fn):
  arena ← _ARENAS.get(pool)；若有：occupied ← arena 这个 segment 里当前非空闲的块（torch.cuda.memory._snapshot()）
  # 1. record 遍：扔掉的 CUDAGraph(keep_graph=True)、自己的 pool、普通 cudaMalloc
  tmo_trace_begin(); fn() 捕获; events = tmo_copy_trace()
  # 2. 规划（障碍 occupied 当作永久存活、位置固定的 buffer）
  plan = greedy_by_size(buffers_from_trace(events), fixed=occupied, ceiling=arena.bytes 或 None)
  graph_record.reset(); empty_cache()                       # 释放 record 遍的 segment
  if 无分配:                                      普通捕获, mode = no_allocations
  elif arena 存在 and plan 跨度 > arena.bytes:    普通捕获, mode = reuse_pool（放不下）
  elif arena 不存在 and 新 arena > record 遍 first-fit 保留量 x (1 + slack): 普通捕获, mode = skipped_no_gain
  # 3. 打包遍
  无 arena：tmo_set_capture_plan(...)，[tmo_va_only_begin()] 新建 arena，[tmo_vmm_commit()]，登记 _ARENAS
  有 arena：tmo_set_capture_plan_at(..., arena.base)，按偏移在已有 arena 的空闲段里切块
  fn() 捕获进真正的 graph
  assert tmo_capture_plan_cursor() == n
```

**为什么每次捕获都要规划，而不只是池上的第一次。** 第一版只规划第一次捕获，后面的都用 first-fit 去填 arena 的空洞。真实 DeepSeek-V3.1 上这样没有收益：arena 是紧的（1874 MiB，活跃峰值 1870 MiB），后面第一次不同 stream 的捕获用 first-fit 照样需要它在新池里需要的 1824 MiB，于是又开了一个 segment，总量回到 3706 MiB（`results/old/` 里留着这次的原始数据，`pool_after` 清楚显示第二个 1824 MiB 的 segment 出现在 group 1 的第一次捕获）。现在每次捕获都做一遍 record + 规划，但打包时不再新建 arena，而是在同一个 arena 里按各自的偏移切块，arena 里只剩一个 segment。代价是每次捕获多一遍 record 遍（约 0.43 s）。

**stream mutex group。** 本仓库使用的 PyTorch 带有 stream mutex group（pdmux 把各 stream group 的 decode stream 登记为一个组，同组的 stream 可以互相复用空闲块并重新打上自己的 stream 标记，见 `get_free_block` 的回退分支）。打包遍里 `try_tmo_malloc` 对覆盖空闲段做同样的事：`cover->stream = stream`，两侧的剩余块继承这个 stream。

**障碍与“顶部区”。** 前面捕获留在 arena 里的块（图的输出）对后面的规划是固定障碍。如果它们落在 arena 中间，会把后面某个大 buffer 需要的连续空间切开。所以 `greedy_by_size` 把“捕获结束仍存活、且很小”（不超过 `max(256 KiB, 总量/256)`）的 buffer 放在 arena 的最高处（在已有 arena 里自顶向下找第一个不冲突的位置，新 arena 里放在其它 buffer 之上），其余 buffer 仍按大小贪心。大的存活 buffer 参与正常贪心，不放顶部区，否则会白白拉长跨度。放不下时降级成 `reuse_pool`，不会出错。

### 5.2 fold：层折叠

CUDA 驱动 API：条件节点 `cuGraphAddNode(CU_GRAPH_NODE_TYPE_CONDITIONAL, WHILE)` 与 `cuGraphConditionalHandleCreate`；设备侧 `cudaGraphSetConditional`、`cudaGraphKernelNodeSetParam`（节点需 `CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE`）；`cuGraphClone / cuGraphNodeFindInClone / cuGraphDestroyNode / cuGraphAddDependencies_v2`；`cuGraphKernelNodeGetParams / cuFuncGetParamInfo / cuKernelGetParamInfo`（读参数布局）；Green Context：`cuStreamGetCtx_v2 + cuCtxFromGreenCtx`、与 context 无关的 `CUkernel`（`cudaGetKernel`）。输入是 `CUDAGraph(keep_graph=True).raw_cuda_graph()` 得到的未实例化 `cudaGraph_t`。

接口（`graph_fold.h`）：`gf_fold(graph, stream, min_iters, min_saved_nodes, dry_run, dump_dir, GfReport*, GfHandle**)`、`gf_launch`、`gf_state`、`gf_destroy`、`gf_dump_graph`。

## 6. 算法

### 6.1 贪心按大小（tensor）

`end = None`（捕获结束仍存活，地址烤进输出）视为 +inf。大小按 allocator 对齐：`round_size` 至少 512 B，向上取 512 的倍数。

```
GreedyBySize(buffers):
    placed = []
    for b in sort(buffers, key = (-size, start, index)):
        cursor = 0
        for o in placed where lifetimes(b, o) overlap, ordered by o.offset:
            if cursor + b.size <= o.offset: break
            cursor = max(cursor, o.offset + o.size)
        b.offset = align_up(cursor, 512)
        placed.append(b)
    return max(b.offset + b.size)           # 规划跨度 packed

arena = arena_size(packed)    # 满足 segment_size(arena) == arena 的最小值：
                              # allocator 按“请求大小”选 pool，再按 get_allocation_size 取整，
                              # arena 必须是这个映射的不动点，否则它会多保留一截（例如 100 KiB 的跨度必须取 20 MiB 而不是 2 MiB）
```

生命周期规则：只有 `free_completed` 结束生命周期，`free_requested` 不结束（跨 stream 使用的块要等 deferred free 完成才可复用）。

### 6.2 折叠（fold）

```
fold(G, S, min_iters, min_saved):
  K ← G 的 kernel 节点（创建顺序）；对每个读取 func/kern、ctx、grid/block/smem、参数布局、launch attributes、参数字节
  cls[k] ← 去掉参数取值后的 launch class
  (S, P, N) ← 周期区间：cls[i] == cls[i+P] 的最长连续段所覆盖的 kernel 数最大；N ≥ min_iters
  region ← K[S .. S+N*P) 的 kernel 与它们之间的 EVENT_RECORD/EMPTY 节点
  若区间内有其它类型的节点（MEMCPY、MEMSET…）：拒绝
  若 展开节点数 - (父图 + body 节点数) < min_saved：拒绝（不值得）
  G' ← clone(G)；删除 region；对被删的 event/empty：前驱直接连后继
  reset ← 新增 gf_reset kernel
  W ← 新增 WHILE，依赖 = 区间的入边来源 ∪ {reset}；W → 区间出边的目标
  body ← 迭代 0 的 P 个 kernel（拷贝全部 launch attribute；PDL 属性除外）+ 迭代内的边 + gf_sched（依赖 body 的汇点）
  差异 ← 各迭代间取值不同的 8 字节字（相邻、间隙 ≤ 1 个干净字的合并为一个更新区间）
  table[i] ← 迭代 i 的更新区间取值（设备内存）；有差异的 body kernel 带 DEVICE_UPDATABLE 属性
  instantiate(G')

gf_sched(...)   # body 最后一个 kernel，单 block 128 线程
  next ← idx + 1；continue ← next < N；row ← continue ? next : 0   # 末次迭代后回到第 0 次迭代的参数
  所有线程并行 cudaGraphKernelNodeSetParam(node_u, offset_u, table[row][u], size_u)
  thread 0：idx ← row；cudaGraphSetConditional(handle, continue)
```

### 6.3 manager 的捕获流程

```
capture(graph, pool, stream, fn, key, group, graph_ctx):
  [tensor] out ← TensorPlanner.capture(...)        失败 -> graph.reset(); 普通捕获（strict 时抛出）
  [无 tensor] out ← 普通捕获
  [fold]  folded ← gf_fold(graph.raw_cuda_graph(), stream, ...)
          成功: folded[key] ← handle（不实例化展开图）；refused: graph.instantiate()（和原来一样）
  [debug] dump
replay(key, graph): folded.launch(current_stream) 或 graph.replay()
```

## 7. 正确性保证

**tensor**

1. *不重叠*：两个生命周期相交的 buffer 地址一定不相交（贪心只在已放置且相交的 buffer 之外寻找空隙），`verify_plan` 逐项检查边界、512 对齐和不重叠（单元测试里对 400 条随机轨迹做 fuzz，debug 模式下对每次真实规划再检查一遍）。
2. *复用的合法性*：只复用 `free_completed` 之后的地址。这与原 allocator 允许复用的条件一致：`free_completed` 表示该块在所有 stream 上的使用都已在捕获图里被当前捕获 stream 的后续节点所依赖。所以任何晚于它捕获的 kernel 都在图的依赖上排在这些使用之后，回放时不会写到仍被读的数据。输出张量（`end=None`）永不与别的 buffer 共用地址。
3. *同一份 kernel 序列*：打包遍与 record 遍运行同一个 `fn`；allocator 在每次分配时检查舍入大小等于计划大小，超出计划数则报错，结束时检查 `cursor == n`。任何不一致都变成显式的异常，由 manager 回退到普通捕获，不会带着错位的计划继续。
4. *数值*：kernel 序列、参数（除指针外）、依赖完全不变；偏移 512 对齐，与 allocator 的保证相同（依赖指针对齐选择向量化路径的 kernel 行为相同）。
5. *同一 arena 里的多张图*：后来的图只在 arena 的空闲段里切块（`cover` 必须是当前空闲块，否则 `TORCH_CHECK` 失败，回退普通捕获），并把前面图仍占用的块（从 allocator 的 snapshot 读出，不是自己记账）当作固定障碍，所以不会和前一张图的输出重叠。各图的临时 buffer 只在各自的回放期间使用，图之间回放串行（同一个 pool 的图本来就不能并发回放，这是 PyTorch 私有池共享的前提）。
6. *Green Context*：不创建 stream，不改 SM 分区；arena 在整个设备地址空间，不按 stream 划分。

**fold**

1. *序更强*：WHILE 在每次迭代结束时有屏障，迭代 i+1 在迭代 i 的全部 kernel 之后开始。原图的跨迭代依赖都是 i -> i+1 方向（遇到反向边则拒绝），所以任何原图允许的顺序，折叠图都满足；不会出现新的竞态。
2. *参数逐字节一致*：每个迭代的参数由原图读出，差异字存表，由 `gf_sched` 在上一迭代结束时写入；常量字保持节点里的值。最后一次迭代后写回第 0 次的参数，下次回放从头开始。
3. *event 节点*：区间内的 EVENT_RECORD/EMPTY（NCCL strongstream 的串行化 event）被摘除，前驱直连后继，kernel 间的次序由依赖边保留；区间外的 EVENT_WAIT/RECORD 保留。
4. *PDL*：programmatic 边变成完整边，PDL 属性去掉；序更强，不改变结果。带 `PROGRAMMATIC_EVENT`、`LAUNCH_COMPLETION_EVENT` 或不能复制的非默认属性则拒绝。
5. *拒绝即回退*：找不到周期区间、区间内有 MEMCPY 等 body 不接受的节点、收益太小，都回到普通 `instantiate`。
6. *Green Context*：WHILE 与 body 使用捕获 kernel 的 ctx（stream 经 `cuStreamGetCtx_v2/cuCtxFromGreenCtx` 解析），`gf_sched/gf_reset` 用与 context 无关的 `CUkernel`，SM 分区保持（单元测试用 `%smid` 验证）。
7. *已知限制*：折叠图的 `launch` 不经过 `CUDAGraph.replay()` 的 RNG 序言（philox offset 不会更新）；带 PyTorch 随机数 kernel 的图不应折叠（SGLang 解码图里没有）。

## 8. 改动清单

新增：

| 文件 | 内容 |
| --- | --- |
| `cuda_graph_memopt/` | 工具本体（第 1 节目录） |
| `cuda_graph_memopt/patches/pytorch-c10-tmo.patch` | 只改 PyTorch `c10/cuda/CUDACachingAllocator.cpp`；已提交到 PyTorch fork |
| `cuda_graph_memopt/tests/` | CPU 单元测试 15 项、GPU 单元矩阵、Green Context 矩阵 |
| `python/sglang/srt/model_executor/cgmo_hook.py` | `cgmo_requested()`、`create_graph_memopt()`、`uses_vmm_arena()`；必要时把仓库根加入 `sys.path` |

修改（相对合并前的 `sglang_prefill_miss`）：

| 文件 | 改动 |
| --- | --- |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | `__init__`：`self.cgmo = self._create_graph_memopt()`；新增 `_create_graph_memopt`（仅基类版本的 runner 启用）；`_create_device_graph` 在启用时返回 `opt.new_graph()`；`_capture_graph` 在启用时转进 `opt.capture(...)`（`key/group` 由 `capture_one_batch_size` 设置，`graph_ctx` 保留 memory-saver 路径）；`capture()` 结束后调用 `verify_pending()` 与 `finish()`；`replay()` 走 `opt.replay(key, graph)`。`CGMO` 未设置时与原代码路径一致 |
| `test_scripts/perf/start_server_pdmux_special_dpa.sh` | CGMO 非空时检查 `cuda_graph_memopt/lib/libc10_cuda.so` 并设置 `LD_LIBRARY_PATH` |
| `python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` | 在 VMM arena 生效时自动设置 `SGLANG_MEMORY_SAVER_CUDA_GRAPH=1`（custom allreduce 用 `cudaMalloc` scratch，IPC 句柄不指向 `cuMemMap` 地址） |

## 9. 复现

```bash
# GPU 单元测试与 Green Context（1 GPU，hi launch 里执行）；输出目录由 CGMO_TEST_OUT 指定
export CGMO_TEST_OUT=bench_out/cgmo_merge/results
python3 cuda_graph_memopt/tests/test_cpu.py                           # CPU 测试
CUDA_VISIBLE_DEVICES=0 bash cuda_graph_memopt/tests/run_unit.sh && python3 cuda_graph_memopt/tests/summarize_unit.py
CUDA_VISIBLE_DEVICES=0 bash cuda_graph_memopt/tests/run_green.sh
# 集成（8 GPU）：合并树的验收矩阵（GSM8K 全量 + Mooncake 前 1000 条，attn_done / oar_done x CGMO 开关）
bash bench_out/cgmo_merge/run_matrix.sh
```

合并进 `sglang_prefill_miss` 之后的验收数据见 `bench_out/cgmo_merge/MERGE_RESULTS.md`。
