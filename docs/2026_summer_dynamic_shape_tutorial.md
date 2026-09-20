# ONNX 动态 Shape 子图项目记录

## 目标

这项工作把 ONNX 的动态输入维度和运行时 Shape Tensor 接入 InfiniTensor 的图、算子、内存规划和 Runtime。整体链路是：

```text
ONNX 输入 [N, 2, 3]
 -> Shape
 -> Gather(batch)
 -> Unsqueeze
 -> Concat([N], [6])
 -> Reshape(data, runtime_shape)
 -> 输出 [N, 6]
```

## 已完成的基础链路

- Python 导入器保存固定维度、符号维度和未知维度的约束信息。
- `set_input()` 检查 rank、正数维度和固定维度，动态维度允许变化。
- `Shape` 输出固定为 `INT64`。
- CPU 注册 `Shape`、`Gather` 和 `Cast`；`Concat` 支持 `INT64` shape tensor。
- `Reshape` 保留旧的静态构造函数，并增加运行时 shape tensor 输入。
- Runtime 增加 shape preparation 阶段：先执行 shape-producing operators，再调用 shape inference 和 `dataMalloc()`，最后运行数据图。
- CUDA 已注册 Shape、Int64 Gather/Concat 路径，并沿用现有 graph capture 状态失效机制；CUDA 动态端到端仍需真实设备验证。
- 静态 shape 子图可在导入阶段做受限常量折叠，并记录折叠节点数。

## 关键 C++ 接口

```cpp
graph->prepareDynamicShapes();
graph->shape_infer();
graph->dataMalloc();
runtime->run(graph);
```

动态 Reshape 的形状值从第二个 Tensor 输入读取，支持 `INT32/INT64`、一个 `-1` 自动推导和 `0` 复制输入维度；元素总数不一致时拒绝执行。

## Python 用法

```python
stub = OnnxStub(onnx_model, backend.cpu_runtime())
for batch in (1, 2, 8, 3, 1):
    stub.set_input([[batch, 2, 3]])
    stub.inputs["x"].copyin_numpy(values)
    stub.run()
    print(stub.getShape("y"))
```

`OnnxStub` 会在 `set_input()` 中完成约束检查、shape tensor preparation、图级 shape inference 和动态内存重规划；调用方不需要重新加载模型。

## 验证

纯 C++ CPU 测试：

```bash
cmake -S . -B build/dynamic-shape-cpu \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DUSE_BACKTRACE=OFF -DBUILD_TEST=ON
cmake --build build/dynamic-shape-cpu -j1 --target test_dynamic_shape
ctest --test-dir build/dynamic-shape-cpu -R '^test_dynamic_shape$' --output-on-failure
```

Python ONNX 测试：

```bash
python3 pyinfinitensor/tests/test_onnxstub.py
python3 pyinfinitensor/tests/test_onnx.py
```

动态测试连续执行 `1 -> 2 -> 8 -> 3 -> 1`，比较输出 shape 和 reshape 后的数值；如果安装了 `onnxruntime`，测试还会比较 ORT 的输出。当前 Python 测试依赖生成的 `backend` 扩展和 `onnx/onnxsim`，缺少依赖时应记录为环境缺口，不把导入失败写成算子失败。

## 优秀项和边界

导入阶段只折叠全部输入 shape 已知的 Shape 子图；只要链路依赖动态输入，就保留运行时节点。这样不会把动态 shape 错误地固定为构建时的占位值。

当前限制：更复杂真实网络模型和完整 DDP/PP 训练黑盒验证尚未完成。CUDA latency A/B benchmark 已完成基础版本；动态 `ConstantOfShape`、动态 axes、`Reshape allowzero`、动态 H/W/sequence、CUDA Graph 多 shape、基础容量复用和高 rank Gather 已有测试证据，双 GPU NCCL 及双进程动态 Shape 集成也已验证。

## AI 编译器原理：从模型到执行

InfiniTensor 将 ONNX 模型转换为内部计算图，再由 Runtime 选择后端 kernel 执行。调用链是：

```text
ONNX protobuf -> Python importer -> GraphHandler -> Graph/Tensor/Operator
             -> shape inference -> memory planner -> KernelRegistry -> CPU/CUDA kernel
```

前端负责把 ONNX 语义转换成内部对象，并检查固定维度约束；Graph 保存 Tensor、Operator 以及 producer/consumer 关系；Operator 负责 `inferShape()`、dtype 推断和 workload key；`dataMalloc()` 根据当前 shape 和 Tensor 生命周期安排 activation storage；Runtime 最后按 `(Device, OpType)` 查找 kernel。新增算子要把这些环节都接上，不能只补一个 kernel。

动态 Shape 的难点是形状值本身也是图计算结果。普通数据 kernel 需要先知道输出大小，但 `Shape -> Gather -> Unsqueeze -> Concat` 的结果又决定 `Reshape` 输出大小，所以执行顺序必须是：

```text
具体输入 shape -> Shape Tensor 子图 -> shape_infer -> dataMalloc -> 完整数据图
```

`getDynamicShapeOperators()` 从动态 Reshape 的第二输入反向收集依赖，只执行相关的 Shape/Gather/Unsqueeze/Squeeze/Concat/Cast，不会误执行无关的数据算子。

## WSL 与镜像依赖复现

```powershell
wsl.exe -d Ubuntu-24.04 --exec /bin/echo WSL_READY
Get-Service wslservice,LxssManager
sc.exe queryex wslservice
```

本次 WSL 故障是 `wslservice` 卡在 `STOP_PENDING`。确认 `sc.exe queryex` 给出的 PID 确实是 `wslservice` 后，可在管理员 PowerShell 中执行：

```powershell
Stop-Process -Id <wslservice-PID> -Force
Start-Service wslservice
wsl.exe -d Ubuntu-24.04 --exec /bin/echo WSL_READY
```

第三方依赖默认使用 GitHub SSH URL。网络受限时，在仓库本机配置镜像：

```powershell
$repo = 'F:\InfiniTensor大模型与人工智能系统训练营\InfiniTensor'
git -C $repo config submodule.3rd-party/backward-cpp.url https://ghfast.top/https://github.com/bombela/backward-cpp.git
git -C $repo config submodule.3rd-party/googletest.url https://gitee.com/mirrors/googletest.git
git -C $repo config submodule.3rd-party/nlohmann_json_cmake_fetchcontent.url https://ghfast.top/https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent.git
git -C $repo config submodule.3rd-party/pybind11.url https://gitee.com/mirrors/pybind11.git
git -C $repo submodule update --init --recursive
```

## 编译、测试与故障复盘

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor" && cmake -S . -B build/dynamic-shape-cpu -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF -DUSE_BACKTRACE=OFF -DBUILD_TEST=ON && cmake --build build/dynamic-shape-cpu -j1'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor" && ctest --test-dir build/dynamic-shape-cpu -R "^test_dynamic_shape$" --output-on-failure'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor" && ctest --test-dir build/dynamic-shape-cpu --output-on-failure'
```

CUDA（本机 RTX 3050、CUDA 12.0、GCC 12.4）复现命令：

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor" && cmake -S . -B build/dynamic-shape-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=ON -DUSE_BACKTRACE=OFF -DBUILD_TEST=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 -DCMAKE_CXX_FLAGS="-I/mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/include -Wno-error=array-bounds" -DCMAKE_CUDA_FLAGS="-I/mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/include" -DCMAKE_EXE_LINKER_FLAGS="-L/tmp/infinitensor-cudnn/lib -L/mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/lib -Wl,-rpath,/mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/lib"'
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd "/mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor" && LIBRARY_PATH=/tmp/infinitensor-cudnn/lib:/mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/lib cmake --build build/dynamic-shape-cuda --target test_dynamic_shape -j2 && LD_LIBRARY_PATH=/usr/lib/wsl/lib:/lib/x86_64-linux-gnu:/mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor/build/dynamic-shape-cuda ctest --test-dir build/dynamic-shape-cuda -R "^test_dynamic_shape$" --output-on-failure'
```

CUDA 12.0 只接受 GCC 12 以内的 host compiler；当前 wheel 版 cuDNN 只带 `libcudnn.so.9`，构建前创建本机临时 linker 名：

```bash
mkdir -p /tmp/infinitensor-cudnn/lib
ln -sfn /mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9 /tmp/infinitensor-cudnn/lib/libcudnn.so
```

项目的 CMake 默认保留 `70;80` 架构兼容性，但现在支持通过 `INFINI_CUDA_ARCHITECTURES` 指定目标架构。双 4090 服务器使用 `-DINFINI_CUDA_ARCHITECTURES=89`，构建命令中实际出现 `compute_89,code=[compute_89,sm_89]`，避免为旧架构重复编译。

当前 CUDA 动态测试实际结果为 `DynamicShape.* 6/6 passed`，包括普通 CUDA 运行、连续 shape 重规划、CUDA Graph 多 shape 重捕获/缓存复用、缩小后复用同一 activation storage、rank-8 Gather 和标量 Gather。

本次实际结果是动态 Shape `1/1 passed`，CPU 全量回归 `42/42 passed`。首次专项失败发生在正式 Runtime 的第一个 `Shape` 算子，最终根因是 `ShapeObj` 未实现 `getWorkloadVector()` 和 `getOpAttrVector()`，生成 `OpPerfKey` 时触发基类 `IT_TODO_HALT()`；补齐性能键接口后，未改变 kernel 数学逻辑即通过。

Python/ORT 复现：

```bash
cd /mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor
make install-python
python3 pyinfinitensor/tests/test_onnxstub.py
python3 pyinfinitensor/tests/test_onnx.py
```

判定标准是同一模型连续运行至少五组 shape，输入变大和变小均成功，输出 shape 和数据正确；只有安装 `onnxruntime` 并实际打印对齐结果后，才能声称完成 ORT 对齐。

## 验收边界

已验证：CPU 动态 Shape 链、单卡 CUDA 动态 Shape 链、动态 Reshape `0/-1/allowzero` 语义、连续 shape 内存重规划、CUDA Graph 多 shape 重捕获/缓存、动态 H/W、sequence 和 Conv 五组 ORT 对齐、动态 `Unsqueeze/Squeeze` axes、静态和动态 `ConstantOfShape`、静态 folding、CPU/CUDA 专项回归、容量 storage 复用以及 CPU/CUDA P50/P99 benchmark、rank-8 Gather 和标量 Gather。双 GPU NCCL、双进程动态 Shape 集成和两进程动态 TP smoke 也已通过。尚未声称通过：更复杂真实网络模型和完整 DDP/PP 训练黑盒验证；已有 NCCL/TP 证据不能替代这两类训练验证。

## 高级场景复现

动态 axes 的 ONNX 形式是 `Unsqueeze(data, axes_tensor)` 或 `Squeeze(data, axes_tensor)`。axes Tensor 可以是模型输入，也可以由 Shape 子图产生；调用 `set_input()` 后 Runtime 先准备 shape value，再执行 `shape_infer()` 和内存规划。测试中同一个模型把 axes 从 `[1]` 改为 `[0]`，输出分别为 `[2,1,3]` 和 `[1,2,3]`。

`Reshape(allowzero=1)` 中，shape Tensor 的 `0` 保持字面零，不再复制输入维度；`allowzero=1` 与 `-1` 同时出现会被拒绝，因为两种规则无法同时推断同一个元素数。默认 `allowzero=0` 仍按 ONNX 复制输入维度。

静态 `ConstantOfShape` 在导入阶段读取常量 shape 和可选 scalar value，生成 initializer，避免运行时执行。动态版本先执行 shape 输入生产者，再由 `ConstantOfShapeObj::inferShape()` 从 INT32/INT64 shape Tensor 读取维度，完成 `shape_infer()` 和内存规划后，最后由 CPU/CUDA fill kernel 写入 scalar value；当前支持 Float32、Int32、Int64 输出。

动态 sequence 测试构造 `[1,S,4] -> Reshape([1,S,4]) -> MatMul(4,4)`，连续运行 `S=1,7,3,12,1`，每组都比较输出 shape、数据和 ONNX Runtime 输出。

动态图像模型测试构造 `[1,1,H,W] -> Conv(3x3,padding=1) -> Relu`，连续使用 `(H,W)=(5,5),(7,6),(3,8),(5,5)`，比较 InfiniTensor 与 ONNX Runtime 的输出 shape 和数值。该测试验证动态 Shape 能驱动真实数据算子重新规划输出内存。

容量 benchmark：

```bash
python3 scripts/benchmark_dynamic_shape.py --warmup 10 --repeat 100
```

一次 CPU 实际结果：same-shape preparation P50/P99=`0.0128/0.0227 ms`、total=`0.0136/0.0251 ms`；alternating-shape preparation=`0.0132/0.0324 ms`、total=`0.0140/0.0339 ms`。CUDA 运行命令：

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/lib/x86_64-linux-gnu:/mnt/d/MLIR/deps/triton-venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/mnt/f/InfiniTensor大模型与人工智能系统训练营/InfiniTensor/build/dynamic-shape-cuda
python3 scripts/benchmark_dynamic_shape.py --device cuda --warmup 10 --repeat 100
```

RTX 3050 的一次实际 CUDA 结果：same-shape preparation P50/P99=`0.1119/0.2763 ms`、total=`0.1992/0.8999 ms`；alternating-shape preparation=`0.1103/0.3035 ms`、total=`0.2003/0.9291 ms`。这些是当前 WSL 单卡环境的观测值，不代表多卡吞吐。

## 还缺哪些验证

补充：双进程 NCCL 动态 Shape 集成已经在双 4090 服务器通过；仍未完成的是完整 DDP/PP 训练黑盒验证。

当前仍未完成的是更复杂真实网络模型和多进程 DDP/PP 黑盒验证。双 4090 服务器已完成双线程 NCCL 动态 Shape 集成测试，但该测试不等同于多进程 DDP。当前本地 WSL 只暴露一张 RTX 3050，因此没有用单卡结果替代服务器上的双 GPU 证据。

双 GPU 服务器复现：

```bash
cd ~/work/InfiniTensor
export CUDNN_HOME=$HOME/qwen-serving/venv/lib/python3.12/site-packages/nvidia/cudnn
export NCCL_HOME=$HOME/qwen-serving/venv/lib/python3.12/site-packages/nvidia/nccl
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:$CUDNN_HOME/lib:$NCCL_HOME/lib:$PWD/build/ai-shape-dist-2gpu:$HOME/work/cudnn-link/lib:$HOME/work/nccl-link/lib:${LD_LIBRARY_PATH:-}
CUDA_VISIBLE_DEVICES=0,1 ./build/ai-shape-dist-2gpu/test_nccl_comm --gtest_filter="NCCL.*"
CUDA_VISIBLE_DEVICES=0,1 ./build/ai-shape-dist-2gpu/test_dynamic_shape_dist --gtest_filter="NCCL.*"
CUDA_VISIBLE_DEVICES=0,1 ./build/ai-shape-dist-2gpu/test_dynamic_shape_multiprocess --gtest_filter="NCCL.dynamic_shape_runs_in_two_processes"
```

实际结果：`NCCL.multi_gpu_communication`、`NCCL.dynamic_shape_runs_on_both_gpus`、`NCCL.dynamic_shape_runs_in_two_processes` 和两进程 dynamic TP smoke 均通过；其中 C++ 双进程测试在 rendezvous 修复后重新编译并复验通过。服务器测试使用 CUDA 12.8、GCC 11.5、CMake 4.4、NCCL wheel；由于远端不能直连 GitHub，源码和本机 submodule 依赖通过 SSH 传输到 `~/work/InfiniTensor`。

TP smoke 的复现命令：

```bash
export PYTHONPATH=$PWD/pyinfinitensor/src:$PWD/examples/distributed:$PWD
CUDA_VISIBLE_DEVICES=0,1 python3 scripts/run_dynamic_tp_smoke.py
```

该脚本建立一个 `[batch,4]` 的动态 Gemm 模型，两个 Python 进程分别运行 TP rank 0/1，并连续执行 `batch=1,5,2,8,1`，同时覆盖普通 Gemm 和 `transB=1`。权重沿输出列切分，AllGather 后恢复 `[batch,4]` 输出，再与单卡基线比较。初次失败的根因是 `parallel.py::shard_tensor()` 没有按传入 `dim` 构造切片，而是固定使用 axis 0；之后还发现两个 worker 都默认绑定 device 0，以及 NCCL rendezvous 文件存在 rank 0 提前删除竞态。修复后输出为 `dynamic TP smoke: 2 processes, 2 GPUs, 5 shapes, normal/transposed Gemm allclose PASS`。

## 问题记录

早期实现把 `dim_param` 和未知维度都替换成 `1`，导致固定/动态维度无法区分；现在由 Python importer 保存约束，Tensor 仍只保存当前具体 shape。另一处问题是 Reshape 只接受静态 initializer；现在增加双输入 operator。Shape 原来没有 runtime kernel，且默认继承输入 dtype；现在有 CPU/CUDA 注册和固定 `INT64` 输出。Shape 相关节点必须在内存重新规划前执行，否则下游输出可能继续使用旧容量。
