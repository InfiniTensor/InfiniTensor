# 分布式脚本

#### 1. 运行pytorch模型并生成输入和标准输出，可选择导出onnx

使用 `--export_onnx` 设置导出onnx的目录，默认为当前路径 `./`，不使用这个flag则只进行计算和生成输入输出。

```bash
python run_pytorch.py --model gpt2  --batch_size 1  --length 1 --export_onnx ./
```

会在当前目录下生成输入输出文件`test_inputs.npy` 和 `test_results.npy`，目前只支持单一输入输出。

#### 2. 运行InfiniTensor分布式脚本

```bash
python cuda_launch.py --model "/XXX/XXX.onnx" --nproc_per_node 4 
```

#### 3. 运行InfiniTensor分布式脚本，测试算子性能

```bash
nsys profile python single_cuda_launch.py --model "/XXX/XXX.onnx" 
```
