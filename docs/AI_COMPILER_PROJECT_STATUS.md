# AI 编译器方向状态与验收矩阵

## 当前里程碑

本仓库当前正在实现 ONNX 动态 Shape 子图编译与执行支持。CPU 和单卡 CUDA 第一条垂直切片已经完成真实构建和运行验证；ORT、动态模型和容量复用仍按证据单独记录，不以代码存在替代测试通过。

## 验收矩阵

| Gate | Required evidence | Current evidence | Status | Gap / next action |
| --- | --- | --- | --- | --- |
| 固定/符号/未知维度表示 | importer 保存约束，固定维度非法变化被拒绝 | `_shape_spec()`、`set_input()` 校验和 Python 单测已写入 | partial | 运行完整 Python suite |
| Shape Tensor 链 | Shape -> Gather -> Unsqueeze -> Concat | CPU/CUDA C++ 链路；动态测试 `6/6 passed`，含 rank-8 和标量 Gather | verified CPU/CUDA | 真实模型级扩展 |
| 动态 Reshape | shape tensor 驱动输出 shape，支持 0/-1/allowzero | 双输入 `ReshapeObj`、importer 分支、溢出/边界检查；动态 batch/HW/sequence 与 `allowzero` 测试通过 | verified CPU/CUDA | 增加更多 rank/非法输入 |
| 动态内存 | 变大/变小、权重不破坏、旧数据不泄漏 | 复用 allocator；CPU 全量 `42/42 passed`，CUDA 动态 C++ 与 Python 通过 | verified CPU/CUDA | 加更大 shape |
| CPU ORT 对齐 | 5 组 shape 的 shape/value allclose | dynamic batch/HW/sequence/Conv/ConstantOfShape 均与 ORT 对齐；`test_onnxstub.py 34/34`（CPU backend，CUDA 用例单独验证） | verified CPU | 固定版本报告 |
| CUDA 动态 Shape | CUDA shape phase、重规划、连续 shape | CUDA 构建成功；`DynamicShape.*` 6/6 passed，另有动态 `ConstantOfShape` CUDA Python 用例通过 | verified CUDA | 增加真实 ONNX 动态模型 |
| 常量折叠优秀项 | 静态 shape 子图节点减少，动态节点不误折叠 | 非拓扑静态 Shape 链折叠、静态 `ConstantOfShape` 测试通过；动态 ConstantOfShape CPU/CUDA 运行通过 | verified basic | 增加更多 ONNX 算子 |
| 动态 H/W 或序列优秀项 | 真实动态模型连续 5 shape + ORT | 动态 H/W 与 `Reshape+MatMul` sequence 链均连续 5 组并与 ORT 对齐 | verified H/W/sequence | 真实网络模型级扩展 |
| CUDA Graph 多 shape 优秀项 | shape/storage 改变重捕获，相同状态复用 | `CudaGraphRecapturesAcrossDynamicShapes` passed；shape `1,2,8,3,5,1`，重复 shape 使用缓存 | verified basic | 加 capture/cache benchmark |
| 容量复用优秀项 | allocation count/peak/latency A/B | 动态测试验证同一 `storageId`；CPU/CUDA 各 100 次 benchmark 已采集 P50/P99 | verified reuse/latency | 扩展更大 shape |

## 架构边界

| Boundary | Owner | Allowed dependencies | Forbidden dependencies | Enforcement |
| --- | --- | --- | --- | --- |
| ONNX importer | `pyinfinitensor/onnx.py` | ONNX protobuf、backend GraphHandler、shape metadata | 直接操作 runtime blob 或私有 allocator | Python importer tests |
| Shape operator contract | `include/operators/*` | Tensor shape/dtype、operator graph | Python-only shape mutation | C++ operator tests |
| Shape execution | `src/kernels/cpu` / `src/kernels/cuda` | Kernel registry、Tensor data、runtime stream | 改变 graph topology | CPU/CUDA kernel tests |
| Shape transaction | `GraphObj` / `RuntimeObj` | shape preparation、shape inference、allocator | 数据 kernel 执行中改 layout | dynamic integration test |
| Memory/runtime adapter | `LazyAllocator` / runtime backends | concrete Tensor shape、storage state | 复制 initializer 到临时 CPU 作为替代实现 | allocator/capture tests |

## 当前风险

- WSL 服务曾处于 `STOP_PENDING`；已终止明确的 `wslservice` PID、重启服务并恢复发行版。
- CUDA `Shape` 当前通过固定的最多 8 个 kernel 参数传递维度，避免每次临时 device allocation；仍需要设备端连续 shape benchmark。
- 当前 CUDA 验证使用 RTX 3050、CUDA 12.0、GCC 12.4；CUDA 配置通过 D 盘 Python wheel 中的 cuDNN 9 头文件和库完成，构建时需使用 `LIBRARY_PATH` 解决 wheel 缺少无版本 `libcudnn.so` 的开发包链接名。
- CMake 已支持 `INFINI_CUDA_ARCHITECTURES`；双 4090 服务器使用 `89`，并通过构建命令确认生成 `compute_89/sm_89`。
- 动态 shape 子图从动态 Reshape 的 shape 输入反向收集，不能按算子类型全图执行；CPU/CUDA Runtime 共用 Graph 的收集结果。
- 当前范围尚未覆盖更复杂真实网络模型和完整 DDP/PP 训练黑盒验证；双 GPU NCCL、双进程动态 Shape 及项目现有 TP launcher smoke 已验证。

## 可复现证据

```text
cmake --build build/dynamic-shape-cpu -j1                 PASS
ctest -R '^test_dynamic_shape$' --output-on-failure       1/1 PASS
ctest --output-on-failure                                42/42 PASS
cmake --build build/dynamic-shape-cuda --target test_dynamic_shape -j2 PASS
DynamicShape.*                                             6/6 PASS
test_onnxstub.py                                           34/34 PASS (CPU backend; CUDA ConstantOfShape separately PASS)
test_onnx.py                                               58/58 PASS
NCCL.multi_gpu_communication                                PASS (2 GPUs)
NCCL.dynamic_shape_runs_on_both_gpus                        PASS (2 GPUs)
NCCL.dynamic_shape_runs_in_two_processes                    PASS (2 processes, 2 GPUs)
dynamic TP smoke                                            PASS (2 Python processes, 2 GPUs, 5 shapes, normal/transposed Gemm allclose)
```

本轮新增证据：动态 `Unsqueeze/Squeeze` axes、`allowzero`、动态 sequence `Reshape+MatMul`、静态和动态 `ConstantOfShape` 均通过；动态 `ConstantOfShape` 的 shape 输入覆盖模型输入和 `Shape` 子图两种路径。CPU benchmark（100 次样本）结果：same-shape preparation P50/P99=`0.0128/0.0227 ms`，total P50/P99=`0.0136/0.0251 ms`；alternating-shape preparation P50/P99=`0.0132/0.0324 ms`，total P50/P99=`0.0140/0.0339 ms`。CUDA benchmark（RTX 3050，100 次样本，64 MiB workspace）结果：same-shape preparation P50/P99=`0.1119/0.2763 ms`，total P50/P99=`0.1992/0.8999 ms`；alternating-shape preparation P50/P99=`0.1103/0.3035 ms`，total P50/P99=`0.2003/0.9291 ms`。

多卡验证当前受硬件限制：本机仅暴露 1 张 RTX 3050，无法诚实执行 DDP/多 GPU CUDA Graph 黑盒测试；该项保持未验证，不能用单卡结果替代。

专项第一次失败的根因是 `ShapeObj` 没有实现 `getWorkloadVector()` 和 `getOpAttrVector()`。Runtime 在执行 Shape 前生成 `OpPerfKey`，落入基类 `IT_TODO_HALT()`；补齐接口后无需改变 kernel 数学逻辑即可通过。

CUDA 专项第一次运行时还遇到两类环境问题：CUDA 12.0 不接受 GCC 13，改用已安装的 GCC 12；RTX 3050 无法承受默认 7 GiB workspace，测试运行时使用 64 MiB 可配置 workspace。之后 `ShapeTensorChainReplansAndPreservesValuesOnCuda` 和 `CudaGraphRecapturesAcrossDynamicShapes` 均通过。

Gather 边界测试补齐了 rank-8 输入和标量 index。前者验证 metadata 数组上限，后者验证 rank-0 index 的输出维度；两项均已通过。初次 rank-8 失败来自测试 oracle 的结果排列错误，修正为逐行 gather 语义后通过。

在双 4090 服务器上，新增的 `NCCL.dynamic_shape_runs_on_both_gpus` 让两个 GPU 分别运行动态 Shape 图，再对各自输出元素数执行 NCCL AllReduce；测试通过，证明双 GPU kernel 执行和 NCCL 通信可以共同工作。该测试采用仓库现有的双线程模型，不等同于多进程 DDP。

进一步新增的 `NCCL.dynamic_shape_runs_in_two_processes` 使用 `fork()+waitpid()` 启动两个独立子进程，每个进程绑定一张 4090、独立创建 InfiniTensor CUDA Runtime/NCCL communicator，并运行动态 Shape 图后执行 AllReduce；在 rendezvous 原子写入和 ready 标记修复后重新编译运行，测试仍通过。它验证了多进程 NCCL 动态 Shape 链路，但仍不等同于完整 DDP/PP 训练黑盒。

项目现有 `examples/distributed/parallel.py` 的两进程 TP smoke 也已通过：同一动态 batch Gemm 模型连续执行 `batch=1,5,2,8,1`，覆盖普通 Gemm 和 `transB=1`，权重沿输出列分片，经 NCCL AllGather 后与单卡基线逐元素 `allclose`。过程中修复了 `shard_tensor()` 忽略 `dim`、始终切分第 0 维的 bug；该 bug 会使 Gemm 权重形状错误并在构图阶段触发 `kA != kB`。另修复了测试 worker 默认都绑定 device 0 的问题，并修复 NCCL 文件 rendezvous 的提前删除竞态。
