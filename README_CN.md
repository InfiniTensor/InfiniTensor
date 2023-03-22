# 使用指南

## 编译

推荐使用 Ubuntu-22.04，本文以此环境为例。

## 环境配置

1. 使用 apt 安装依赖

   > 如果不使用 Ubuntu-22.04，部分软件版本可能不够高。

   ```bash
   sudo apt-get install make cmake build-essential python-is-python3 python-dev-is-python3 python3-pip libdw-dev
   ```

2. 安装 Protobuf

   ```bash
   wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-cpp-3.21.12.tar.gz
   tar xf protobuf-cpp-3.21.12.tar.gz
   cd protobuf-3.21.12
   ./autogen.sh
   ./configure CFLAGS="-fPIC" CXXFLAGS="-fPIC"
   make -j8
   cd protobuf-3.21.12
   sudo make install
   sudo ldconfig
   ```

3. 更新 pip 并换清华源

   ```bash
   python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. 编译并安装 python 库

   > 第一次执行会同时安装 python 依赖库，比较慢

   仅编译 CPU 部分：

   ```bash
   make install-python
   ```

   编译 GPU 部分：

   ```bash
   make install-python CUDA=ON
   ```

## 使用项目

项目管理功能已写到 [Makefile](Makefile)，支持下列功能：

- 编译项目：`make`/`make build`
- 清理生成文件：`make clean`
- 安装 python 库：`make install-python`
- 测试 c++ 后端：`make test-cpp`
- 测试 python 前端：`make test-onnx`

## 基于 python 前端执行推理任务

TODO 详细说明

```python
import os, onnx, unittest, cv2
import numpy as np
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_tensor,
    make_graph,
    make_tensor_value_info,
)
from onnx.checker import check_model
from pyinfinitensor.onnx import from_onnx, backend, cuda_runtime, to_onnx
# prepare mode file and image file
model_file = "path/to/your/onnx file"
image_file = "path/to/your/image"
if model_file != None:
    # load model to cuda runtime
data, output, handler = from_onnx(onnx.load(model_file), cuda_runtime())
# load image to cuda runtime
    image = cv2.imread(image_file)
    image = cv2.resize(image,(224,224))
    image = image.reshape(-1)
handler.copy_float(data["data"],image.tolist())
# run the network
handler.run()
# print the result
    for key,value in output.items():
        vec = value.cloneFloats()
        print(vec)

```
