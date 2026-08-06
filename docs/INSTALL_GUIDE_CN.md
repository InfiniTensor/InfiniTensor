# 安装部署指南

## 环境准备

1. 使用 GCC 11.3 或更高版本，以及 CMake 3.17 或更高版本。
2. 安装 `make`、`build-essential`、Python 3、pip 和 `libdw-dev` 等基础依赖。
3. 仅使用原生 CPU 时，不需要安装设备 SDK。
4. 使用加速卡时，先按设备厂商说明安装并验证驱动与 SDK，再构建匹配该环境的 InfiniRT 和 InfiniOps。

InfiniTensor 不直接包含厂商 SDK 头文件或 runtime 实现。设备支持范围由外部 InfiniRT/InfiniOps 安装前缀决定，因此切换硬件不需要修改 InfiniTensor 源码或启用逐厂商 CMake 开关。

## 编译项目

### 原生 CPU

未指定 `INFINIOPS_ROOT` 时，构建会自动关闭 InfiniOps 执行后端：

```bash
make install-python INFINI=OFF
```

### InfiniOps 执行后端

先分别安装目标环境对应的 InfiniRT 和 InfiniOps，再执行：

```bash
make install-python \
  INFINI=ON \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix
```

若 InfiniOps 已经将匹配的 InfiniRT 安装到同一前缀，可以让两个变量指向同一路径。使用 ATen 实现时，InfiniOps 必须在目标机器上以 `WITH_TORCH=ON` 构建。
若该 InfiniOps 链接 PyTorch，还应设置 `INFINIOPS_CXX11_ABI=0` 或 `1`，
并与 `torch.compiled_with_cxx11_abi()` 的结果保持一致。

分布式构建使用 InfiniCCL：

```bash
make install-python \
  DIST=ON \
  INFINICCL_ROOT=/path/to/infiniccl-prefix \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix
```

不同硬件使用同一条 InfiniTensor 编译命令，只替换外部安装前缀。运行时通过 InfiniRT 提供的设备名称选择目标：

```python
from pyinfinitensor import backend

runtime = backend.runtime("<infini-rt-device>", index=0)
```

`backend.runtime("cpu")` 始终创建 InfiniTensor 原生 CPU runtime；其他设备名称由当前 InfiniRT 构建提供。名称无效或该设备未编译时会返回明确错误。

## 测试

```bash
make test-cpp
make test-onnx
make test-api
```

加速卡验证前应先确认驱动、SDK、InfiniRT 和 InfiniOps 在当前机器上可独立工作。InfiniTensor 不负责安装或修改这些外部组件。

## Docker

```bash
make docker-build
make docker-run
make docker-exec
```

目标镜像需要自行提供对应的驱动运行环境、SDK 以及匹配的 InfiniRT/InfiniOps 安装前缀。

## 技术支持

如遇到问题，请联系我们技术支持团队。
