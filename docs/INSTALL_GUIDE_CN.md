# 安装部署指南

## 前置条件

InfiniTensor 负责计算图、ONNX 前端和执行调度。加速设备的算子实现与运行时由 InfiniOps 和 InfiniRT 提供。使用加速设备前，先在目标环境中完成与该设备匹配的 InfiniRT 和 InfiniOps 安装；InfiniTensor 不直接依赖厂商 SDK。

需要准备：

- CMake 3.17 或更高版本；
- 可用的 C/C++ 编译器、Python 3、pip 和 make；
- 构建加速后端时，使用与 InfiniOps 相同的 Python/PyTorch 及 C++11 ABI；
- 驱动、厂商 SDK、InfiniRT 和 InfiniOps 已在目标机器上完成各自的基础验证。

使用昇腾 ATen provider 的 C++ 宿主需要先用匹配环境的 Python executable 配置解释器并导入 `torch`、`torch_npu`，且调用线程可以取得 GIL；Python 接口会加载 `INFINIOPS_PROVIDER_MODULES` 指定的模块。未初始化的 C++ 调用会明确报错。独立 C++ 测试使用 CMake 选定的 Python 来初始化环境，以保持标准库、site-packages 和厂商编译子进程的一致性；测试运行时仍需指定对应 provider 模块。这些要求不适用于纯原生 provider。

运行旧 IR 的 RoPE 时，安装的 Ops SDK 需要导出 `RotaryEmbedding`；不再依赖已弃用的 `RotaryEmbeddingInfinilm`。按设备与 SDK 能力选择已有原生或 ATen 实现，均通过同一个公开接口调用。

普通 Release 构建默认 `BACKTRACE=OFF`，不需要 `libdw-dev`。只有启用 backtrace 时才需要该系统依赖。

## 获取源码

GitHub 操作使用 SSH：

```bash
git clone git@github.com:InfiniTensor/InfiniTensor.git
cd InfiniTensor
git submodule update --init --recursive
```

`3rd-party/InfiniCore` 统一固定下层组件版本，InfiniOps、InfiniRT 和 InfiniCCL
源码位于它的 `submodules/` 目录。它们不会在构建 InfiniTensor 时自动编译。
下层构建参数、厂商 SDK 和设备架构参数仍在各自仓库维护；当前单设备执行不需要构建 InfiniCCL。

先构建 InfiniRT，再将 InfiniOps 安装到同一前缀。以下为不依赖 PyTorch 的 CPU 示例：

```bash
export INFINI_PREFIX="$HOME/.infini"
cmake -S 3rd-party/InfiniCore/submodules/InfiniRT -B build/InfiniRT \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON \
  -DCMAKE_INSTALL_PREFIX="$INFINI_PREFIX"
cmake --build build/InfiniRT --parallel
cmake --install build/InfiniRT

cmake -S 3rd-party/InfiniCore/submodules/InfiniOps -B build/InfiniOps \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON -DWITH_TORCH=OFF \
  -DGENERATE_PYTHON_BINDINGS=OFF -DINFINI_RT_ROOT="$INFINI_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$INFINI_PREFIX"
cmake --build build/InfiniOps --parallel
cmake --install build/InfiniOps

make install-python INFINI=ON ATEN=OFF \
  INFINIOPS_ROOT="$INFINI_PREFIX" INFINIRT_ROOT="$INFINI_PREFIX" \
  PYTHON="$(command -v python3)"
```

加速设备和 ATen provider 的 SDK、构建选项、算子选择及 Python 依赖，分别参见
`3rd-party/InfiniCore/submodules/InfiniRT/README.md` 和
`3rd-party/InfiniCore/submodules/InfiniOps/README.md`。

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

### 当前组件组合的能力边界

组件版本由 InfiniCore 子模块引用固定。依赖更新应按 InfiniOps、InfiniCore、InfiniTensor 的顺序合入；本分支不依赖历史开发版 InfiniOps 的
`BeginGraphCaptureMemory` 和 value-only pooling helper 不再是构建前提。
普通 MaxPool 使用标准 `MaxPool2dWithIndices` 接口；非标准索引格式的 provider
需要独立适配和验证，不能沿用历史 Ascend pooling 的通过记录。

图捕获、实例化、重放和销毁统一通过 InfiniRT，不依赖旧的 InfiniOps 捕获内存接口。
InfiniTensor 在捕获前创建实际选中的算子实例，检查其 `capture_safe()`，并将实例
和稳定工作区保留到图销毁。未声明安全的实现（包括当前未标记的 ATen provider）
在进入捕获前明确拒绝，普通执行仍可用，不以普通前向替代图重放。
同一 ATen 构建中的安全原生实现仍可捕获，不因启用 ATen 而整体关闭 Graph。

Ascend / MLU 的 provider 初始化、外部张量与流互操作也须按正式组件版本重新验证；
旧开发版的跨平台测试和图捕获结果不适用于这个依赖组合。
`INFINITENSOR_REQUIRE_GRAPH_CAPTURE=1` 仍用于要求图能力的专项测试，能力缺失时会失败。

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

### 检索组件更新与开发验证

检索所需的 CPU/NVIDIA/MLU/昇腾算子由 InfiniOps 提供。模型输入的 shape 推导由前端负责，算子实现的参数约束由 InfiniOps 的公开工厂检查，通用执行器不维护 RMSNorm 等算子的专用兜底。

正式构建使用已提交的 InfiniCore 引用；开发阶段可通过 `INFINIOPS_ROOT` 和 `INFINIRT_ROOT` 指向独立安装前缀，验证候选组件，无需为每次试验更新依赖引用。应记录实际源码和二进制摘要。启用昇腾 ATen provider 时，启动 Python 前设置 `TASK_QUEUE_ENABLE=0` 和 `INFINIOPS_PROVIDER_MODULES=torch_npu`，确保与原生 runtime stream 的提交顺序一致。
