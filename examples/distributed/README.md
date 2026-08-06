# 分布式示例

分布式执行统一使用 InfiniRT runtime 和 InfiniCCL，不再维护逐平台启动脚本。

```bash
python launch.py \
  --device nvidia \
  --model /path/to/model.onnx \
  --nproc-per-node 4
```

`--device` 接受当前 InfiniRT 安装所提供的设备名称。切换硬件时只需要更换目标机器上的 InfiniRT、InfiniOps 和 InfiniCCL 安装，以及命令行设备名称，不需要修改 InfiniTensor 源码。

启动脚本会为每个 rank 创建独立 runtime，并通过统一的 `runtime.init_comm()` 初始化通信。运行前需要准备与模型匹配的输入和参考输出文件。
