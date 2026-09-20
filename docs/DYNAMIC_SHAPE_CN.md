# ONNX 动态 Shape 子图编译与执行支持

本次实现与验证基于 `5cb7c1fc9d01e53ace9013667295307f13907e7a`。WSL 目录：`/root/onnx-dynamic-shape/InfiniTensor1`；独立环境：`/root/onnx-dynamic-shape/.venv`。

## 快速复现

在 Windows 终端进入 WSL：

```powershell
wsl -d Ubuntu-22.04-D -u root
```

已有环境直接执行：

```bash
cd /root/onnx-dynamic-shape/InfiniTensor1
source /root/onnx-dynamic-shape/.venv/bin/activate
make test-dynamic-shape
make demo-dynamic-shape
make benchmark-dynamic-shape
python -m pytest -q pyinfinitensor/tests/test_onnxstub.py pyinfinitensor/tests/test_onnx.py pyinfinitensor/tests/test_api.py
```

从新环境部署（Ubuntu 22.04，GCC >=11.3，CMake >=3.17）：

```bash
sudo apt-get update
sudo apt-get install -y git make cmake build-essential python3-dev python3-venv python3-pip libdw-dev
# 在已应用本次补丁的仓库根目录运行；仅本次命令使用镜像，不修改全局配置。
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash examples/dynamic_shape/setup.sh
```

setup.sh 使用 HTTPS 临时覆盖公开子模块的 SSH 地址，创建独立虚拟环境并以 2 个并行任务编译，避免本机 6.7 GiB WSL 内存不足。CPU 构建按安装指南等价执行 CMake、复制 backend 模块和 editable 安装；没有修改用户默认 WSL 用户或系统 Python 包。

## 使用接口

```python
import onnx
import numpy as np
from pyinfinitensor import backend
from pyinfinitensor.onnx import DynamicOnnxStub

model = onnx.load('artifacts/dynamic_shape/images.onnx')  # 只加载一次
session = DynamicOnnxStub(model, backend.cpu_runtime())
for batch in [1, 2, 8, 3, 1]:
    result = session.run({'x': np.ones((batch, 3, 8, 8), dtype=np.float32)})
    print(result['y'].shape)
print(session.input_specs)
print(session.stats)
print(session.handler.memory_stats())
```

必须使用显式 `DynamicOnnxStub.run(feeds)` 入口；原 `OnnxStub` 保持静态接口兼容，不会自动改用新动态执行器。第一次 run 构造数据图，之后保留所有 C++ Tensor/Operator 对象，不重新加载 ONNX 或重建数据图。输出为独立 NumPy 拷贝，可安全跨多次推理保留。

## 设计与实现

1. ONNX 输入声明以 int / str / None 分别保留固定、符号和匿名未知维度；输入的实际形状单独绑定。验证输入集合、rank、dtype、固定维度以及同名符号一致性。
2. Shape 子图编译为主机控制程序：Shape、Gather、Unsqueeze、Squeeze、Concat、Cast、Constant。控制值使用 NumPy Tensor，Shape 读取元数据而非整个数据 Tensor。初始化器或常量节点的纯静态依赖在导入时折叠，运行时依赖每次重新计算。
3. 在整个数据图内存规划前，拓扑求值 Shape 程序并检查 Reshape 目标。支持目标为图输入 Tensor，支持 0 复制维度和单个 -1 推导；检查 dtype、rank、元素总数和非法维度。
4. C++ Reshape 增加目标维度更新，Operator 绑定增加单节点形状推导，复用各数据算子的 inferShape。主机计划与 C++ 推导结果一致后调用原 Graph.dataMalloc，随后写入输入并运行所选 InfiniTensor CPU 或 CUDA Kernel。
5. 权重首次分配后只复制一次。沿用仓库已有的权重池与动态激活池；新增统计接口暴露实际激活池分配次数、当前容量、当前规划需求和权重字节数。
6. 只求值 Shape 控制程序，不使用 ONNX Runtime 替代数据推理。ONNX Runtime 仅在测试与 Demo 中作为独立数值对照。

本方案是“主机 Shape 程序 + 持久 C++ 数据图”。Shape 节点不作为 C++ Runtime Kernel 执行，也未把运行时目标作为 C++ Reshape 的第二条图边；它由调度器求值后更新算子目标参数。该设计覆盖本项目模型，不能视为通用设备侧 Shape Tensor 执行支持。

## 已验证结果

- Python：98 passed，2 skipped，11 subtests passed；其中新增动态测试 18 项。
- C++：41 个测试目标全部通过，含新增 Reshape 连续更新和内存验证。
- 动态图片：`[1,3,8,8] -> [2,3,12,10] -> [8,3,16,12] -> [3,3,10,8] -> [1,3,8,8]`；ORT 最大绝对误差 0。
- 序列投影：`[1,3,4] -> [2,8,4] -> [8,16,4] -> [3,5,4] -> [1,3,4]`；含 MatMul、Relu 和两次 Reshape；ORT 最大绝对误差 2.3841858e-7。
- 对齐标准：输出形状完全一致，float32 使用 rtol=1e-4、atol=1e-5；控制 Tensor 验证 int64/int32 的 dtype 和整数值。
- Shape 折叠：图片模型控制节点从 5 个减为 4 个；关闭折叠的结果同样与 ORT 一致。
- 容量复用：100 次推理（预热后）实际新增激活池分配 0 次，对照每次 trim 为 80 次。两组最大已提交激活池容量均为 98304 字节。中位端到端时间分别为 0.315133 ms、0.2967385 ms；本次未体现速度提升。

上述为初始 CPU 构建记录；后续 CUDA 构建的完整回归为 **107 passed、13 subtests passed，无跳过**，C++ **78/78** 目标通过。

## CUDA 部署与第三项优秀指标

本机实测：WSL Ubuntu 22.04、GTX 1650 4 GiB（sm_75）、Windows 驱动 572.83、nvcc 12.8.93、cuDNN 8.9.7.29。使用 Windows 提供的 WSL GPU 驱动，不安装 Linux 显卡驱动。

首次安装 Toolkit 开发组件（需 root，网络可访问 NVIDIA 软件源）：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-libraries-dev-12-8 cuda-sanitizer-12-8 libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2
# 先按前文 setup.sh 准备基础依赖、子模块和 venv，再构建 CUDA：
CUDA_ARCHS=75 bash examples/dynamic_shape/setup_cuda.sh
```

已有环境运行及完整验收：

```bash
cd /root/onnx-dynamic-shape/InfiniTensor1
source /root/onnx-dynamic-shape/.venv/bin/activate
source examples/dynamic_shape/env_cuda.sh
python examples/dynamic_shape/cuda_demo.py
bash examples/dynamic_shape/validate_cuda.sh
```

接口改为 `DynamicOnnxStub(model, backend.CudaRuntime(workspace_size=64*1024*1024))`。Shape 控制仍在主机执行，float32 数据计算实际使用 InfiniTensor CUDA，不借助 ORT 推理。默认 Runtime 工作区从固定 7 GiB 改为可配置的 512 MiB，Demo 使用 64 MiB。算子请求超出配置时会报错，应按模型需求增大工作区；原 Attention 的硬编码 2 GiB 临时缓冲区已改为按缓存 Shape 计算，并验证跨 16-token 分块。

同一实例的 CUDA 图片输入为 `[1,3,8,8] → [2,3,12,10] → [8,3,64,48] → [3,3,10,8] → [1,3,8,8]`；序列为 `[1,3,4] → [2,8,4] → [8,64,4] → [3,5,4] → [1,3,4]`。每组同时检查 CPU、ORT、输出 Shape、持久 Tensor/Operator 身份和权重内容。图片最大误差 0，序列相对 CPU/ORT 最大绝对误差均为 2.3841858e-7，容差 rtol=1e-4、atol=1e-5。

图片激活池容量（字节）为 `3072 → 12288 → 1179648 → 1179648 → 1179648`，累计分配次数为 `1 → 2 → 3 → 3 → 3`：增大时扩容，缩小时复用且输出无残留。权重池保持 256 字节。统计不包含 Runtime 工作区、CUDA 库内部缓冲区和切换时瞬时峰值。

Compute Sanitizer 2025.1 对池化与朴素分配两种模式均报告 `ERROR SUMMARY: 0 errors`、`LEAK SUMMARY: 0 bytes leaked in 0 allocations`。命令保留完整 memcheck/leak-check，使用 `--show-backtrace no`：本机开启回溯时观察到工具保留 Python validate 栈帧与 GPU 对象，导致退出泄漏报告；关闭回溯后正常释放。没有调用 cudaDeviceReset 掩盖未释放对象，也未禁用 Kernel 检测。参数含义见 [NVIDIA Compute Sanitizer 文档](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

证据：`cuda-results.json`、`cuda-naive-results.json`、`cuda-sanitized-results.json`、`cuda-naive-sanitized-results.json`、`cuda-full-python-tests.log`、`cuda-cpp-tests.log`、`cuda-memcheck.log`、`cuda-naive-memcheck.log`。均位于 `artifacts/dynamic_shape/`。memcheck 结论仅覆盖这些模型、形状和 GPU；不能推广为任意 ONNX 模型保证。

## 优秀指标与边界

| 方向 | 结论 |
| --- | --- |
| Shape 子图编译期优化 | 已完成常量折叠，提供折叠前后节点数及正确性对照；未实现符号表达式的部分求值。 |
| 真实动态 H/W 或序列模型 | 已完成动态 H/W 人工模型和轻量序列投影网络的五组验证；没有训练/预训练真实业务模型，严格“真实模型”指标仅部分完成。 |
| CUDA 动态 Shape | 已完成第三项：连续五组变形、扩容/缩容、权重保护、CPU/ORT 对齐、两种分配器 memcheck 无错误及无泄漏。支持范围如下。 |
| CUDA Graph 集成 | 原有 C++ CUDA Graph 回归通过；尚未对本项目 ONNX 动态 Demo 完整验收捕获/复用指标，不计为完成第四项。 |
| 容量复用与性能 | 已集成并验证仓库已有复用机制，新增可复现统计基准；分配减少有证据，延迟提升和进程瞬时峰值内存没有证据。 |

支持标准 ONNX opset 13-18、已知输入 rank、正维度。Shape 控制算子只支持项目所需子集；Gather 等要求操作数在控制程序中可求值。数据图支持 float32 Reshape、Identity、Relu、Add/Sub/Mul、二维 MatMul、Transpose。控制程序依赖任意数据计算值、未知 rank、空维度、Reshape allowzero=1、ConstantOfShape、Expand、控制流及其他数据算子会报错或不在支持范围。不支持对同一实例并发 run。

动态参数错误会在执行 Kernel 前报 ValueError；内存不足可减小输入；新增算子超出白名单会给出明确异常。不应在修改计算图后直接调用低级 handler.run 绕过动态规划。

基准统计的是已提交激活池容量，不包含分配切换瞬间旧池与新池同时存活的峰值、Python/ORT 内存及 CUDA 显存；对照是显式 trim 策略，并非另一个旧版本二进制。时延包含主机形状规划、内存规划、输入输出拷贝和 Kernel；环境抖动与小模型开销不可忽略。

## 交付与证据

`artifacts/dynamic_shape/` 内保存 results.json、memory_benchmark.json、python-tests.log、cpp-tests.log、environment.txt 和生成的 ONNX 模型。模型可由 Demo 重新生成；不要在补丁中添加二进制构建产物。

报告提交人尚未提供，PDF 保留“待填写”。本地尚未提交 Git Commit、推送或创建上游 PR：WSL 尚无 Git 作者配置和 GitHub CLI 登录。已准备 PR 正文与完整补丁供评审；本次未发送任何邮件。

参考仓库文档：
- https://github.com/wjia8591-dou/InfiniTensor1/blob/master/docs/INSTALL_GUIDE_CN.md
- https://github.com/wjia8591-dou/InfiniTensor1/blob/master/docs/USER_GUIDE_CN.md
- https://github.com/wjia8591-dou/InfiniTensor1/blob/master/docs/SUPPORT_MATRIX_CN.md
