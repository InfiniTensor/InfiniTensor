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

## 分布式构建

设置 `DIST=ON` 并提供 InfiniCCL 前缀：

```bash
make build \
  INFINI=ON \
  DIST=ON \
  INFINIOPS_ROOT="$INFINIOPS_ROOT" \
  INFINIRT_ROOT="$INFINIRT_ROOT" \
  INFINICCL_ROOT=/path/to/infiniccl-prefix
```

多卡通信配置、设备可见性和启动参数由 InfiniCCL 与目标运行时负责。分布式示例见 `examples/distributed/README.md`。

## 常见问题

- `InfiniOps` 或 `InfiniRT` 未找到：检查两个前缀及其 CMake 包是否完整。
- Python/PyTorch ABI 不一致：使用与 InfiniOps ATen 构建相同的 Python/PyTorch 环境重新构建。
- provider 导入失败：先加载该后端要求的 Python provider，并确认其动态库搜索路径。
- 设备 runtime 创建失败：检查设备名称、可见设备编号、驱动和下层运行时安装。
