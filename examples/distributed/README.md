# 分布式脚本

## 英伟达平台运行方式

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

## 寒武纪平台运行方式

**将上述运行脚本 `run_pytorch.py` 以及 `cuda_launch.py` 针对寒武纪平台做了相应的适配，具体见 `run_pytorch_mlu.py` 以及 `bang_launch.py`。**

#### 1. 运行pytorch模型并生成输入和标准输出，可选择导出onnx

使用 `--export_onnx` 设置导出onnx的目录，默认为当前路径 `./`，不使用这个flag则只进行计算和生成输入输出。

```bash
python run_pytorch_mlu.py --model gpt2  --batch_size 1  --length 1 --export_onnx ./
```

会在当前目录下生成输入输出文件`test_inputs.npy` 和 `test_results.npy`，目前只支持单一输入输出。

#### 2. 运行InfiniTensor分布式脚本

```bash
python bang_launch.py --model "/XXX/XXX.onnx" --nproc_per_node 4 
```