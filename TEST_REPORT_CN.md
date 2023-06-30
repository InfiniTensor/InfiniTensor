# TensorSense 适配寒武纪 MLU370X4 测试报告

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
| Densenet121 | Cambricon MLU |  Yes        |
| Densenet201 | Cambricon MLU |  Yes        |
| InceptionV3 | Cambricon MLU |  Yes        |
| InceptionV4 | Cambricon MLU |  Yes        |

## 性能测试

性能测试方法：使用本项目进行万次端到端运行，并进行时间测量，获得万次平均值。

目前各个支持的网络模型性能如下

TensorSense 未开启优化：本项目（ Driver v5.10.10 ）关闭编译优化开关
TensorSense 开启优化：本项目（ Driver v5.10.10 ）打开编译优化开关
Pytorch-MLU：寒武纪官方提供的在其 MLU 加速卡上的 Pytorch 拓展（ Cambricon Driver v5.10.10；v1.10.0-torch1.9-ubuntu20.04-py37 ）

| Model       | Input Size    | Device                  |  TensorSense 未开启优化 |  TensorSense 开启优化   | Pytorch-MLU             |
| --------    | ------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| Resnet18    | (1,3,32,32)   | Cambricon MLU370x4 单卡 |  0.002465 s             |  0.001516 s             |  0.003768 s             |
| Resnet34    | (1,3,32,32)   | Cambricon MLU370x4 单卡 |  0.004486 s             |  0.002734 s             |  0.006425 s             |
| Resnet18    | (1,3,224,224) | Cambricon MLU370x4 单卡 |  0.010648 s             |  0.006910 s             |  0.006939 s             |
| Resnet34    | (1,3,224,224) | Cambricon MLU370x4 单卡 |  0.018839 s             |  0.012550 s             |  0.013070 s             |
| Resnet18    | (16,3,224,224)| Cambricon MLU370x4 单卡 |  0.111386 s             |  0.063843 s             |  0.065089 s             |
| Resnet34    | (16,3,224,224)| Cambricon MLU370x4 单卡 |  0.201821 s             |  0.122844 s             |  0.126135 s             |
| Densenet121 | (1,3,32,32)   | Cambricon MLU370X4 单卡 |  0.008808 s             |  0.005354 s             |  0.025662 s             |
| Densenet201 | (1,3,32,32)   | Cambricon MLU370X4 单卡 |  0.015687 s             |  0.009698 s             |  0.043055 s             |
| Densenet121 | (16,3,32,32)  | Cambricon MLU370X4 单卡 |  0.023589 s             |  0.011230 s             |  0.023029 s             |
| Densenet201 | (16,3,32,32)  | Cambricon MLU370X4 单卡 |  0.040871 s             |  0.020281 s             |  0.038030 s             |
| InceptionV3 | (1,3,32,32)   | Cambricon MLU370X4 单卡 |  0.016028 s             |  0.012421 s             |  0.015799 s             |
| InceptionV3 | (16,3,32,32)  | Cambricon MLU370X4 单卡 |  0.087216 s             |  0.072448 s             |  0.017555 s             |
| InceptionV4 | (1,3,32,32)   | Cambricon MLU370X4 单卡 |  0.033413 s             |  0.026998 s             |  0.023601 s             |
| InceptionV4 | (16,3,32,32)  | Cambricon MLU370X4 单卡 |  0.275981 s             |  0.249290 s             |  0.034467 s             |

从数据可以看出，在国产设备寒武纪加速卡上，当本项目开启了编译优化，网络耗时较未开启编译优化时大幅度减少，性能大幅度提升。

同时，本项目开启编译优化时，大部分网络的性能已经远超于业界主流编程框架产品 Pytorch 。

本项目持续开发中，后续我们会持续针对硬件设备特点进行定制优化，进一步提升网络性能。

## 技术支持

如若您遇到了本项目的问题，请联系我们的技术支持团队
