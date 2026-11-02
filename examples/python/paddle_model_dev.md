# Paddle 模型示例

这些脚本展示从 CIFAR-10 训练、Paddle 模型导出到 InfiniTensor 推理的流程，分别使用 ResNet、DenseNet 和 Inception。

## 环境

准备支持训练设备的 Paddle 环境，以及其 ONNX 导出依赖、OpenCV 和已安装的 `pyinfinitensor`。数据集首次运行会下载到 Paddle 缓存，训练时间和内存需求由模型与 batch 配置决定。

## 运行

从本目录执行一个脚本：

```bash
python paddle_resnet.py
# 或 python paddle_densenet.py
# 或 python paddle_inception.py
```

脚本训练并评估模型，将 ONNX 文件写入 `onnx.save/`，再使用 InfiniTensor 执行测试集推理。

## 设备选择

Paddle 训练设备由脚本中的 `paddle.device.set_device("gpu")` 选择；InfiniTensor 推理设备由 `INFINITENSOR_DEVICE` 选择，默认 `cpu`。两者是独立配置：

```bash
INFINITENSOR_DEVICE=nvidia python paddle_resnet.py
```

更换训练设备需要相应 Paddle 后端；更换推理设备需要匹配的 InfiniOps／InfiniRT 环境，见[安装指南](../../docs/INSTALL_GUIDE_CN.md)。模型导出的节点和形状仍需满足[模型运行接口](../../docs/USER_GUIDE_CN.md)的要求。
