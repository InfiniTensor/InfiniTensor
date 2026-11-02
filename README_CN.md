# InfiniTensor

[English](README.md) | [文档导航](docs/INDEX.md)

InfiniTensor 是计算图推理引擎，提供 ONNX 导入、图变换和 C++／Python 执行接口。它既可以使用内置 CPU runtime，也可以通过 InfiniOps 和 InfiniRT 在加速设备上执行单设备计算图。

## 主要能力

- **运行 ONNX 模型**：导入模型、设置输入、执行计算图和读取输出，也可将图重新导出为 ONNX。
- **直接使用 InfiniOps 算子**：从 SDK 公开接口生成绑定，以统一入口构图；已支持的参数类型无需为每个算子编写后端包装。
- **复用执行资源**：准备并复用算子实例和工作区；设备及算子实现支持时，可通过 InfiniRT 捕获和重放计算图。

InfiniTensor 负责前端语义、图与调度，InfiniOps 负责算子实现，InfiniRT 负责设备、内存、流和设备图。模型的可用算子、dtype 和形状范围取决于前端覆盖及所选后端的实现。

## 安装

获取源码后初始化子模块：

```bash
git submodule update --init --recursive
```

仅使用内置 CPU runtime：

```bash
make install-python INFINI=OFF PYTHON="$(command -v python3)"
```

使用 InfiniOps 后端时，先安装匹配的 InfiniRT 和 InfiniOps，再构建：

```bash
make install-python INFINI=ON \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix \
  CHECK_BACKEND=infini \
  PYTHON="$(command -v python3)"
```

原生 provider 使用 `ATEN=OFF`；依赖 ATen 的 provider 使用 `ATEN=ON`，并准备匹配的 Python/PyTorch 环境。

## 文档

- [安装指南](docs/INSTALL_GUIDE_CN.md)：依赖、构建、provider 配置与安装自检。
- [使用指南](docs/USER_GUIDE_CN.md)：设备选择、ONNX 推理、直接构图与图重放。
- [开发指南](docs/DEVELOPMENT_CN.md)：组件边界、扩展方式、测试组织与 CI。

项目同时保留 [EinNet](https://github.com/InfiniTensor/InfiniTensor/tree/NNET_e2e) 和 [PET](https://github.com/thu-pacman/PET) 等计算图与张量程序优化研究工作，论文引用见 [英文 README](README.md#reference)。
