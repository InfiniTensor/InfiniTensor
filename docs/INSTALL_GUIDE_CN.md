# 安装部署指南

## 前置条件

InfiniTensor 负责计算图、ONNX 前端和执行调度。加速设备的算子实现与运行时由 InfiniOps 和 InfiniRT 提供。使用加速设备前，先在目标环境中完成与该设备匹配的 InfiniRT 和 InfiniOps 安装；InfiniTensor 不直接依赖厂商 SDK。

需要准备：

- CMake 3.17 或更高版本；
- 可用的 C/C++ 编译器、Python 3、pip 和 make；
- 构建加速后端时，使用与 InfiniOps 相同的 Python/PyTorch 及 C++11 ABI；
- 驱动、厂商 SDK、InfiniRT 和 InfiniOps 已在目标机器上完成各自的基础验证。

普通 Release 构建默认 `BACKTRACE=OFF`，不需要 `libdw-dev`。只有启用 backtrace 时才需要该系统依赖。

## 获取源码

GitHub 操作使用 SSH：

```bash
git clone git@github.com:InfiniTensor/InfiniTensor.git
cd InfiniTensor
git submodule update --init --recursive
```

`3rd-party/InfiniOps` 和 `3rd-party/InfiniRT` 固定当前版本对应的下层源码版本。它们不会在构建 InfiniTensor 时自动编译。下层构建参数、厂商 SDK 和设备架构参数应留在各自仓库中维护。

## 仅使用 CPU

```bash
make install-python \
  INFINI=OFF \
  PYTHON="$(command -v python3)"
```

## 使用加速设备

先设置已经安装好的、彼此兼容的 InfiniOps 和 InfiniRT 前缀：

```bash
export INFINIOPS_ROOT=/path/to/infiniops-prefix
export INFINIRT_ROOT=/path/to/infinirt-prefix
```

然后构建并安装：

```bash
make install-python \
  INFINI=ON \
  ATEN=ON \
  INFINIOPS_ROOT="$INFINIOPS_ROOT" \
  INFINIRT_ROOT="$INFINIRT_ROOT" \
  PYTHON="$(command -v python3)"
```

如果 InfiniOps 后端依赖 Python provider，在安装和运行时使用同一组模块：

```bash
make install-python \
  INFINI=ON \
  ATEN=ON \
  INFINIOPS_ROOT="$INFINIOPS_ROOT" \
  INFINIRT_ROOT="$INFINIRT_ROOT" \
  PROVIDER_MODULES=provider_module \
  PYTHON="$(command -v python3)"

export INFINIOPS_PROVIDER_MODULES=provider_module
```

`INFINIOPS_ROOT` 和 `INFINIRT_ROOT` 必须来自同一套兼容构建。ATen 版本的 InfiniOps 必须以 `WITH_TORCH=ON` 构建，并与当前 Python/PyTorch 使用相同的 C++11 ABI。若 provider 动态库不在 Python/PyTorch 环境的默认搜索路径中，可设置 `INFINIOPS_PROVIDER_LIBRARY_DIRS`。

Ascend ATen 构建还需要与 PyTorch 匹配、提供
`c10_npu::getStreamFromExternal` 的 torch_npu。对于 PyTorch 2.7.1，使用
torch_npu 2.7.1.post10；原始 2.7.1 wheel 缺少此接口，会在 InfiniOps
配置阶段被拒绝。InfiniOps 使用 NPU 专用 `from_blob` 封装 InfiniRT 分配的
内存，并将生成的 ATen 算子绑定到 InfiniRT 的流。

必须在导入 torch_npu 之前设置 `TASK_QUEUE_ENABLE=0`。当前 torch_npu 的
主机异步任务队列无法保证外部流上 ATen 算子与 InfiniRT 复制、捕获结束之间
的提交顺序；启用该队列会被明确拒绝。此设置关闭主机侧的延迟提交，设备流
仍可异步执行，图捕获与重放仍然可用。

运行时先导入 `torch_npu` 并初始化 NPU；使用 Python 包安装入口时，将
`PROVIDER_MODULES` / `INFINIOPS_PROVIDER_MODULES` 设置为 `torch_npu`。
在加载 CANN 环境后，可对构建目录执行以下普通运行及捕获重放回归：

```bash
TASK_QUEUE_ENABLE=0 PYTHONPATH=/path/to/infinitensor-build:$PYTHONPATH \
  python test/infini/test_ascend_aten.py
```

原生构建不依赖 PyTorch 或上述 NPU provider。可运行
`bash scripts/test_infiniops_native_build.sh`，从固定子模块提交创建全新的
CPU provider 源码、构建及安装目录，验证 `USE_INFINIOPS_ATEN_KERNELS=OFF`。

## 验证安装

先确认 Python 包可以导入：

```bash
python3 -c 'import pyinfinitensor; import backend; print(pyinfinitensor.__file__); print(backend.__file__)'
```

CPU 可用时可创建 CPU runtime：

```bash
python3 - <<'PY'
from pyinfinitensor import backend
print(backend.runtime("cpu", 0))
PY
```

加速设备名称由 InfiniRT 决定。使用目标环境实际提供的名称和可见设备编号创建 runtime；名称无效、设备不可见或下层后端未构建时会直接报错，不会回退到 CPU。

项目内测试入口：

```bash
make test-cpp
make test-onnx
make test-api
```

## 分布式执行

当前通用 Infini 后端只支持单设备执行。`BUILD_DIST=ON` 会在 CMake 配置阶段明确报错。分布式通信将在后续 InfiniCCL 集成中单独实现和验证。

## 常见问题

- `InfiniOps` 或 `InfiniRT` 未找到：检查两个前缀及其 CMake 包是否完整。
- Python/PyTorch ABI 不一致：使用与 InfiniOps ATen 构建相同的 Python/PyTorch 环境重新构建。
- provider 导入失败：先加载该后端要求的 Python provider，并确认其动态库搜索路径。
- 设备 runtime 创建失败：检查设备名称、可见设备编号、驱动和下层运行时安装。
