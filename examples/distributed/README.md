# 分布式示例

分布式执行使用 InfiniRT 创建各 rank 的 runtime，并通过 InfiniCCL 初始化通信。切换硬件时，只更换目标环境中的 InfiniRT、InfiniOps、InfiniCCL 和设备名称，不需要修改 InfiniTensor 源码。

## 准备数据

假设 `--name demo`，启动目录中需要准备：

- `demo_inputs.npy`：模型输入；
- `demo_results.npy`：参考输出。

文件内容必须与模型的输入输出 shape 和 dtype 一致。

## 启动

```bash
python3 launch.py \
  --device nvidia \
  --model /path/to/model.onnx \
  --name demo \
  --nproc-per-node 4
```

`nvidia` 只是设备名称示例。`--device` 必须使用当前 InfiniRT 构建实际提供的名称。

常用参数：

- `--model`：ONNX 模型路径；
- `--device`：InfiniRT 设备名称；
- `--name`：输入和参考输出文件的前缀，默认 `test`；
- `--nproc-per-node`：当前节点启动的进程数，默认 `1`；
- `--num-nodes`：节点数，默认 `1`；
- `--matmul-compute-type`：矩阵乘计算类型，可选 `default`、`fp16` 或 `tf32`。

启动器为每个本地 rank 创建独立 runtime，并调用 `runtime.init_comm()` 初始化通信。运行前请确认设备数量、InfiniCCL 配置和可见卡号与进程数一致。
