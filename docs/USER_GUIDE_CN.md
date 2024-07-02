# 使用指南

## 目录

- [使用方法](#使用方法)
- [python 前端应用指南](#python-前端应用指南)
  - [导入 onnx 模型](#导入-onnx-模型)
  - [优化](#优化)
  - [导出 onnx 模型](#导出-onnx-模型)
  - [执行推理](#执行推理)
  - [样例代码](#样例代码)
- [技术支持](#技术支持)
- [测试](#测试)

## 使用方法

项目管理功能已写到 [Makefile](../Makefile)，支持下列功能：

- 编译项目：`make`/`make build`
- 清理生成文件：`make clean`
- 安装 python 库：`make install-python`
- 测试 c++ 后端：`make test-cpp`
- 测试 python 前端：`make test-onnx`

并使用下列环境变量传递选项参数：

- `TYPE`：编译模式（`debug`/`release`），默认值为 `release`
- `CUDA`：是否编译 CUDA 后端，默认为 `OFF`，`ON` 打开
- `BANG`：是否编译寒武纪后端，默认为 `OFF`，`ON` 打开
- `KUNLUN`：是否编译昆仑后端，默认为 `OFF`，`ON` 打开
- `ASCEND`：是否编译华为后端，默认为 `OFF`，`ON` 打开
- `BACKTRACE`：是否启用栈回溯，默认为 `ON`，`OFF` 关闭，建议调试时打开
- `TEST`：是否编译 `googletest`，默认为 `ON`，`OFF` 关闭，只有 `test-cpp` 时必要

## python 前端应用指南

`make install-python` 会将项目的 python 前端以 `pyinfinitensor` 为名字安装到系统目录，可以直接 `import pyinfinitensor` 来使用。现阶段，项目的主要用法是从 onnx 导入模型进行优化，然后可以再导出优化后的模型到 onnx，也可以直接运行推理。

### 导入 onnx 模型

支持的模型：

- [x] [ResNet18-v2](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx)
- [x] [DenseNet-121-12](https://github.com/onnx/models/blob/main/validated/vision/classification/densenet-121/model/densenet-12.onnx)
- [x] [Inception-2](https://github.com/onnx/models/blob/main/validated/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx)
- [x] [EfficientNet-Lite4](https://github.com/onnx/models/blob/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx)

```python
import onnx
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor import backend

stub = OnnxStub(onnx.load("model_file"), backend.cpu_runtime())
```

[`onnx.load`](https://onnx.ai/onnx/api/serialization.html#load-a-model) 是 onnx 提供的加载函数，将 onnx 文件读取为保存在内存中的 onnx 模型。

`OnnxStub` 是 onnx 模型在项目中的表示，通过构造这个对象，将 onnx 模型导入到项目中。其构造器的第一个参数是 onnx 模型文件；第二个参数是模型运行的后端运行时，可以是 `backend.cpu_runtime()`、`backend.cuda_runtime()` 或 `backend.bang_runtime()`。

构造出的 stub 对象可以用于操作项目中的模型和运行时。

### 优化

TODO

### 导出 onnx 模型

优化后的模型可以导出成 onnx 文件提供给其他运行时。

```python
with open("optimized.onnx", "wb") as f:
    f.write(stub.to_onnx("optimized").SerializeToString())
```

`stub.to_onnx(<name>)` 将模型转换为 onnx 模型对象，`<name>` 将填写到 onnx 模型的 `name` 字段。序列化到文件的代码见[官方示例](https://onnx.ai/onnx/intro/python.html#model-serialization)。

要可视化检查导出的模型文件，可以利用 [onnx 提供的功能](https://onnx.ai/onnx/api/shape_inference.html#infer-shapes)将所有的张量的形状推理出来再导出：

```python
from onnx.shape_inference import infer_shapes

with open("optimized.onnx", "wb") as f:
    f.write(infer_shapes(stub.to_onnx("optimized")).SerializeToString())
```

然后用 [Netron](https://netron.app/) 绘制计算图。

### 执行推理

也可以使用项目的运行时执行推理。

第一步是将数据传入计算图。`OnnxStub.inputs` 是一个 `Dict[str, Tensor]`，保存着模型的所有输入的名字和对象。可以用 [`items()`](https://docs.python.org/zh-cn/3/library/stdtypes.html#dict.items) 来遍历。

这个代码片段显示了如何打印出模型所有输入张量的名字、形状和对象指针：

```python
for name, tensor in stub.inputs.items():
    print(name, tensor.shape(), tensor)
```

对于 [resnet18-v2-7.onnx](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx)，会打印出：

```plaintext
data [1, 3, 224, 224] <backend.Tensor object at 0x7efeb828e3b0>
```

当然，地址是随机的。这个输出表明需要输入一个名为 “data”，形为 1×3×224×224 的数据。通常来说，这表示一张 224×224 的 rgb 图片。而这个模型是一个 1000 分类的图像分类模型。

为了方便，这里我们向模型传入一个随机的数据。

```python
import numpy

stub.init()
for name, tensor in stub.inputs.items():
    print(name, tensor.shape(), tensor)
    input = numpy.random.random(tensor.shape()).astype(numpy.float32)
    tensor.copyin_float(input.flatten().tolist())
```

`stub.init()` 为所有张量分配空间。空间是预分配的，所以不支持动态 size 的模型。

`tensor.copyin_float(<data>)` 向张量传入数据。其参数必须是一个 `List[float]`，即压平的数据。类似的函数还有 `copyin_int32(<data>)` 和 `copyin_int64(<data>)`

然后，调用 `stub.run()` 执行推理：

```python
stub.run()
```

最后，将结果拷贝出来，传入类似：

```python
stub.init()
for name, tensor in stub.outputs.items():
    print(name, tensor.shape(), tensor)
    print(tensor.copyout_float())
```

### 样例代码

您可以参照[resnet.py](https://github.com/wanghailu0717/NNmodel/blob/main/ResNet/resnet.py)的样例代码进行了解，并尝试运行。在这个文件中，我们使用了 Pytorch 构建了 resnet 网络。您可以查阅该脚本使用方式：

```python
python resnet.py -h
```

在样例代码中，我们对定义的网络进行了序列化操作，并存储为模型文件。之后加载该模型文件，并转换为本项目的模型进行优化操作，再进行推理。您可以关注一下代码中 242 行之后的代码。请注意，您可以按照您的需求来进行操作，通常来说，您所需要撰写的代码就是加载模型，转换为本项目的模型进行优化，推理运行。

## 技术支持

如若您遇到了本项目的问题，请联系我们的技术支持团队

## 测试

除了单元测试 `make test-cpp` 和 `make test-onnx` 之外，还可以用其他方式来测试单个模型导入导出和优化的正确性。

这个脚本利用 onnxruntime 来测试导出的模型是否与导入的模型等价：

```python
import onnx
import numpy
import sys
from onnx import ModelProto, ValueInfoProto
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor import backend
from onnxruntime import InferenceSession


def infer(model: ModelProto, input) -> dict:
    collection = set()
    for node in model.graph.node:
        for output in node.output:
            collection.add(output)
    model.graph.output.extend([ValueInfoProto(name=x) for x in collection])
    session = InferenceSession(model.SerializeToString())
    i = session.get_inputs()[0].name
    return dict(
        zip(
            [x.name for x in session.get_outputs()],
            [x.flatten() for x in session.run(None, {i: input})],
        )
    )


model0 = onnx.load(sys.argv[1])
model1 = OnnxStub(model0, backend.cpu_runtime()).to_onnx("new")

input_shape = [x.dim_value for x in model1.graph.input[0].type.tensor_type.shape.dim]
input = numpy.random.random(input_shape).astype(numpy.float32)

output0 = infer(model0, input)[model0.graph.output[0].name]
output1 = infer(model1, input)[model1.graph.output[0].name]

print("error =", sum((output1 - output0) ** 2) / len(output0))
```

要运行脚本，先安装 onnxruntime：

```bash
pip install onnxruntime
```

打印出的 `error = ...` 是两个模型输出张量的均方误差。对于不同的模型，这个误差最小为 0，最大不超过 1e-9。
