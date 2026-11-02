# 使用指南

安装步骤见[安装指南](INSTALL_GUIDE_CN.md)。以下接口在 Python 中通过 `pyinfinitensor` 使用。

## 选择设备

```python
from pyinfinitensor import backend

runtime = backend.runtime("cpu", index=0)  # 内置 CPU runtime
```

使用 InfiniOps／InfiniRT 的默认设备：

```python
runtime = backend.InfiniRuntime(backend.default_infini_device(), 0)
```

也可通过 `backend.runtime("nvidia", index=0)` 等具体名称选择设备。设备名称由所安装的 InfiniRT 提供；`"cpu"` 入口始终选择内置 CPU runtime。加速后端的 provider 环境需要在导入 Python 包前完成配置。

## 运行 ONNX 模型

为模型准备 `model.onnx` 和保存输入数组的 `inputs.npz`。数组名称对应模型输入名称，shape 与 dtype 对应模型声明：

```python
import numpy as np
import onnx
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub

runtime = backend.runtime("cpu", index=0)
stub = OnnxStub(onnx.load("model.onnx"), runtime)

with np.load("inputs.npz") as inputs:
    for name, tensor in stub.inputs.items():
        tensor.copyin_numpy(np.ascontiguousarray(inputs[name]))

stub.run()
outputs = {name: tensor.copyout_numpy() for name, tensor in stub.outputs.items()}
np.savez("outputs.npz", **outputs)
```

`OnnxStub` 构造时完成图导入和初始化。同一实例可多次写入新输入并调用 `run()`。ONNX 导入器必须能解释模型节点、属性和 shape 参数，目标后端也必须提供相应实现；创建加速 runtime 后，缺失的实现会报错。

调整输入尺寸时，`stub.set_input(shapes)` 按 `stub.inputs` 的顺序接收尺寸，重新推导形状并准备内存。能否调整尺寸取决于图中各节点及 shape 参数的支持范围；它不是通用的符号形状求值接口。更新后应重新写入本次请求的输入。

单输入、静态 shape、FP32 模型也可使用随机输入示例：

```bash
python examples/python/onnx_inference.py model.onnx
```

设置 `INFINITENSOR_DEVICE` 可选择该脚本的设备。该脚本还需安装 PyTorch 以展示输出；实际业务输入应使用上面的数组输入方式并完成预处理。

## 直接构造 InfiniOps 节点

统一 `infini_op()` 入口使用 SDK 公开的算子名称和参数名称。以广播加法为例：

```python
import json
import numpy as np
from onnx import TensorProto
from pyinfinitensor import backend

runtime = backend.InfiniRuntime(backend.default_infini_device(), 0)
graph = backend.GraphHandler(runtime)
x = graph.tensor([2, 4], TensorProto.FLOAT)
bias = graph.tensor([4], TensorProto.FLOAT)
y = graph.tensor([2, 4], TensorProto.FLOAT)
x.set_input()
bias.set_weight()
y.set_output()

schemas = json.loads(backend.infini_operator_schemas())
print(schemas["Add"])
node = graph.infini_op("Add", {"input": [x], "other": [bias]}, {"out": [y]})
graph.data_malloc()
x.copyin_numpy(np.arange(8, dtype=np.float32).reshape(2, 4))
bias.copyin_numpy(np.ones(4, dtype=np.float32))
graph.run()
print(y.copyout_numpy())
```

输入、输出均为参数名称到张量列表的映射；标量属性通过第四个字典传入。支持的属性类型包括整数、浮点数、布尔、字符串、整数列表、DataType 及对应 optional 类型。DataType 可使用 ONNX 类型编号或 InfiniRT 类型名称。可用签名和必需参数以 `infini_operator_schemas()` 为准。

通用节点的输出 shape 和 dtype 由构图方提供。修改输入尺寸后，先更新通用节点的输出尺寸，再推导图形状和分配内存：

```python
graph.change_shape([3, 4], x.fuid())
node.set_output_shapes([[3, 4]])
graph.shape_infer()
graph.data_malloc()
x.copyin_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
bias.copyin_numpy(np.ones(4, dtype=np.float32))
graph.run()
```

## 图捕获与重放

对重复执行的图，可调用 `stub.run_with_graph()` 或 `graph.run_with_graph()`。InfiniTensor 通过 InfiniRT 捕获和重放设备任务，缓存与图结构、尺寸及存储绑定匹配的执行图。

捕获需要运行时提供 Graph API，且每个选中实现声明捕获安全。依赖 host 计算或不支持外部流捕获的路径可能无法捕获；接口会报错。普通执行使用 `run()`。

图缓存持有执行所需的实例和工作区。应用需要释放缓存资源时调用 `runtime.clear_graph_cache()`；该操作会等待已提交的任务，并清理相关准备资源。InfiniOps 后端的实现选择由算子库负责，不支持 `tune()` 调优入口。

## 导出 ONNX

```python
import onnx
from onnx.shape_inference import infer_shapes

model = infer_shapes(stub.to_onnx("exported"))
onnx.checker.check_model(model)
onnx.save(model, "exported.onnx")
```

导出能力取决于节点是否有 ONNX 映射。结构检查与数值检查应分别进行：使用相同输入比较原模型与导出模型的输出，并根据数据类型设置容差。

## 执行范围与问题定位

通用 Infini 后端执行单设备图，分布式构建选项 `BUILD_DIST=ON` 不可用。厂商 provider 的 dtype、布局、外部流与图捕获限制以其实现为准；部分 provider 的内部缓冲独立于 InfiniTensor 的外部共享工作区。

定位失败时，依次检查模型导入、前端语义和形状、算子实现、设备环境。提交问题时附模型或最小构图程序、输入、完整错误信息、运行设备及组件环境，便于复现。
