# 测试报告

## 目录

- [功能测试](#功能测试)
- [性能测试](#性能测试)
- [技术支持](#技术支持)

## 功能测试

目前验证过的网络模型有

| Model       | Device        |  Success    |
| --------    | ------------  | ----------- |
| Resnet18    | Cambricon MLU |  Yes        |
| Resnet34    | Cambricon MLU |  Yes        |
| Resnet50    | Cambricon MLU |  Yes        |
| Resnet101   | Cambricon MLU |  Yes        |
| Resnet152   | Cambricon MLU |  Yes        |
| Densenet121 | Cambricon MLU |  Yes        |
| Densenet169 | Cambricon MLU |  Yes        |
| Densenet201 | Cambricon MLU |  Yes        |
| Densenet161 | Cambricon MLU |  Yes        |
| InceptionV3 | Cambricon MLU |  Yes        |

## 性能测试

性能测试方法：使用本项目进行万次端到端运行，并进行时间测量，获得万次平均值。

目前各个支持的网络模型性能如下

TensorSense：本项目（ Driver v5.10.10 ）
Pytorch-MLU：寒武纪官方提供的在其 MLU 加速卡上的 Pytorch 拓展（ Cambricon Driver v5.10.10；v1.10.0-torch1.9-ubuntu20.04-py37 ）
MagicMind：寒武纪官方提供的推理引擎（ Cambricon Driver v5.10.10；magicmind:1.1.0-x86_64-ubuntu20.04-py_3_7 ）
TensorRT：英伟达提供的推理引擎 （）
Pytorch：既 Meta 提供的开源深度学习编程框架 （）

EndToEnd：既端到端的意思

| Model       | Input Size    | Device                  |  TensorSense EndToEnd   |  Pytorch-MLU EndToEnd   |  MagicMind EndToEnd     |
| --------    | ------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| Resnet18    | (1,3,224,224) | Cambricon MLU370x4 单卡 |  0.005676 s             |  0.005676 s             |  0.005676 s             |
| Resnet34    | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| Resnet50    | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| Resnet101   | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| Resnet152   | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| Densenet121 | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| Densenet169 | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| Densenet201 | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| Densenet161 | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |
| InceptionV3 | (1,3,224,224) | Cambricon MLU370X4 单卡 |                         |                         |                         |


| Model       | Input Size    | Device                  |  TensorSense EndToEnd   |  Pytorch EndToEnd       |  TensorRT EndToEnd      |
| --------    | ------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| Resnet18    | (1,3,224,224) | Nvidia A100 单卡        |  0.005676 s             |  0.005676 s             |  0.005676 s             |
| Resnet34    | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| Resnet50    | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| Resnet101   | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| Resnet152   | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| Densenet121 | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| Densenet169 | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| Densenet201 | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| Densenet161 | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |
| InceptionV3 | (1,3,224,224) | Nvidia A100 单卡        |                         |                         |                         |


## 技术支持

如若您遇到了本项目的问题，请联系我们的技术支持团队
