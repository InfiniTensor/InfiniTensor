# InfiniTensor

[English README](README.md) | [项目文档](docs/INDEX.md)

InfiniTensor 用于导入、变换和执行计算图，可直接运行 ONNX 模型。项目内置 CPU runtime；加速卡上的算子与设备管理分别由 InfiniOps 和 InfiniRT 提供，分布式通信由 InfiniCCL 提供。

## 架构边界

- **InfiniTensor**：负责计算图、ONNX 前端、图变换和执行调度。
- **InfiniOps**：提供统一算子接口及各硬件后端实现。
- **InfiniRT**：提供设备、内存、流和运行时接口。
- **InfiniCCL**：提供多卡和多机通信能力。

InfiniTensor 不包含厂商 SDK 头文件，也不设置逐平台编译分支。目标硬件是否可用，应以该机器上实际安装的 InfiniOps、InfiniRT、驱动和 SDK 为准。

模型支持也不由一张固定清单定义。一个模型能否运行，取决于 ONNX 导入覆盖、算子语义、数据类型与形状，以及目标 InfiniOps 后端是否提供相应实现。缺失算子会直接报错，不会静默回退到 CPU。

## 开始使用

```bash
git submodule update --init --recursive

# 仅使用原生 CPU runtime
make install-python INFINI=OFF PYTHON="$(command -v python3)"

# 使用加速卡；先准备匹配的 InfiniOps 和 InfiniRT 安装前缀
make install-python \
  INFINI=ON \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix \
  PYTHON="$(command -v python3)"
```

详细步骤见：

- [安装部署指南](docs/INSTALL_GUIDE_CN.md)
- [使用指南](docs/USER_GUIDE_CN.md)
- [分布式示例](examples/distributed/README.md)
