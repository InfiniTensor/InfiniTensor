# 支持矩阵

## 目录


- [环境支持](#环境支持)
- [神经网络支持](#神经网络支持)
- [技术支持](#技术支持)

## 环境支持

InfiniTensor 内置 `NativeCpu` 执行路径；加速设备统一通过外部
InfiniRT 和 InfiniOps 安装提供，项目本身不再包含厂商 runtime、SDK
头文件或硬件 kernel。

| Execution provider | Device | Support |
| --- | --- | --- |
| NativeCpu | CPU | Yes |
| Infini | InfiniRT/InfiniOps 当前安装所支持的设备 | Depends on external installation |

NVIDIA GPU 和 Cambricon MLU 已完成端到端模型验证。其他设备的可用性以
对应 InfiniRT/InfiniOps 版本及目标机器验证结果为准。

## 神经网络支持

目前已经验证过的神经网络模型有

- [x] [ResNet18-v2](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx)
- [x] [DenseNet-121-12](https://github.com/onnx/models/blob/main/validated/vision/classification/densenet-121/model/densenet-12.onnx)
- [x] [Inception-2](https://github.com/onnx/models/blob/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx)
- [x] [EfficientNet-Lite4](https://github.com/onnx/models/blob/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx)

## 技术支持

如若您遇到了本项目的问题，请联系我们的技术支持团队
