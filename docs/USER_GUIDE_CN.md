# 使用指南

## 1. 常用构建命令

项目通过顶层 `Makefile` 提供常用入口：

- `make build`：编译 C++ 项目；
- `make clean`：清理生成文件；
- `make install-python`：编译并安装 `pyinfinitensor`；
- `make test-cpp`：运行 C++ 测试；
- `make test-onnx`：运行 ONNX 前端测试；
- `make test-api`：运行 Python API 测试。

主要变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PYTHON` | `python3` | 用于构建和安装的 Python |
| `TYPE` | `Release` | CMake 构建类型 |
| `INFINI` | `ON` | 启用 InfiniOps 执行后端；缺少下层前缀时会关闭 |
| `ATEN` | `ON` | 启用 InfiniOps 生成的 ATen 实现 |
| `INFINIOPS_ROOT` | 空 | InfiniOps 安装前缀 |
| `INFINIRT_ROOT` | 空 | 与 InfiniOps 匹配的 InfiniRT 安装前缀 |
| `BACKTRACE` | `OFF` | 启用栈回溯；需要额外系统依赖 |
| `TEST` | `ON` | 构建测试目标 |
| `PROVIDER_MODULES` | 空 | 安装检查前预加载的 Python provider |

完整安装步骤见[安装部署指南](INSTALL_GUIDE_CN.md)。

## 2. 选择运行设备

原生 CPU 和加速设备使用同一入口：

```python
from pyinfinitensor import backend

cpu_runtime = backend.runtime("cpu", index=0)
device_runtime = backend.runtime("device_name", index=0)
```

`device_name` 由当前 InfiniRT 构建提供，例如某个已启用的 GPU 或 NPU 后端名称。名称无效、设备不可见或后端未编译时会直接报错。

如果 InfiniOps 依赖额外 Python provider，请在启动应用前设置：

```bash
export INFINIOPS_PROVIDER_MODULES=module_a,module_b
```

## 3. 导入并运行 ONNX 模型

下面的示例使用 NumPy 输入执行一个 ONNX 模型：

```python
import numpy as np
import onnx

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub

runtime = backend.runtime("cpu", index=0)
stub = OnnxStub(onnx.load("model.onnx"), runtime)

stub.init()

for name, tensor in stub.inputs.items():
    shape = tuple(tensor.shape())
    value = np.random.random(shape).astype(np.float32)
    tensor.copyin_numpy(value)
    print("input", name, shape, value.dtype)

stub.run()

outputs = {
    name: tensor.copyout_numpy()
    for name, tensor in stub.outputs.items()
}

for name, value in outputs.items():
    print("output", name, value.shape, value.dtype)
```

切换到加速卡时，只替换 runtime：

```python
runtime = backend.runtime("device_name", index=0)
```

输入的 shape 和 dtype 必须与模型声明一致。实际应用应使用预处理后的数据，不要照搬示例中的随机输入。

## 4. 模型兼容范围

InfiniTensor 不用固定模型名单定义支持范围。模型能否运行，需要同时满足：

1. ONNX 图中的节点、属性和张量信息能够被前端导入；
2. 图执行所需的算子语义已经接入 InfiniTensor；
3. 目标 InfiniOps 后端支持相应算子、数据类型和形状；
4. InfiniRT 能够在目标设备上正确创建 runtime、分配内存并执行任务。

因此，某个测试模型通过只能说明该模型及其覆盖路径已验证，不能作为完整支持边界。遇到失败时，应先判断问题属于 ONNX 导入、InfiniTensor 图执行、InfiniOps 算子实现，还是 InfiniRT/驱动环境。

缺失算子或后端实现会明确报错，不会自动转到 CPU 执行。

## 5. 导出 ONNX

`OnnxStub` 可以重新导出计算图：

```python
model = stub.to_onnx("optimized")

with open("optimized.onnx", "wb") as file:
    file.write(model.SerializeToString())
```

需要补充 shape 信息时，可以使用 ONNX 的 shape inference：

```python
from onnx.shape_inference import infer_shapes

model = infer_shapes(stub.to_onnx("optimized"))

with open("optimized.onnx", "wb") as file:
    file.write(model.SerializeToString())
```

导出后可以使用 [ONNX checker](https://onnx.ai/onnx/api/checker.html) 检查结构，并使用 [Netron](https://netron.app/) 查看计算图。数值正确性应使用与原模型相同的输入逐项比较，容差根据模型精度和目标数据类型设定。

## 6. 命令行示例

仓库提供一个通用 ONNX 推理示例：

```bash
python3 examples/python/onnx_inference.py model.onnx
```

默认使用 CPU。通过环境变量选择加速设备：

```bash
INFINITENSOR_DEVICE=device_name \
python3 examples/python/onnx_inference.py model.onnx
```

## 7. 测试与问题定位

安装后先运行与改动最相关的测试：

```bash
make test-cpp
make test-onnx
make test-api
```

模型验证建议同时保存：

- 原始 ONNX 文件及 SHA-256；
- 输入数据、参考输出和生成方式；
- InfiniTensor、InfiniOps、InfiniRT 的版本；
- 驱动、SDK、Python 和 provider 版本；
- 设备名称、卡号、执行命令和数值误差。

提交问题时，请附上最小复现模型、命令、完整错误日志和上述环境信息。问题入口为 [GitHub Issues](https://github.com/InfiniTensor/InfiniTensor/issues)。


## 8. 直接使用 InfiniOps 算子

构建时，InfiniTensor 从已安装 InfiniOps SDK 的公开工厂声明生成绑定。
算子是否可用取决于该 SDK 的构建选择及设备实现，不再需要在 InfiniTensor
新增 kernel 类、工厂映射、dispatch 声明或实现编号。

以下接口直接构造通用节点，参数名称来自 SDK；输入与输出均用名称到张量列表的映射：

```python
import json
from onnx import TensorProto
from pyinfinitensor import backend

runtime = backend.InfiniRuntime(backend.default_infini_device(), 0)
handler = backend.GraphHandler(runtime)
x = handler.tensor([2, 4], TensorProto.FLOAT)
w = handler.tensor([4], TensorProto.FLOAT)
y = handler.tensor([2, 4], TensorProto.FLOAT)
x.set_input()
w.set_weight()
y.set_output()

schemas = json.loads(backend.infini_operator_schemas())
print(schemas["RmsNorm"])
op = handler.infini_op(
    "RmsNorm", {"input": [x], "weight": [w]}, {"out": [y]}, {"eps": 1e-6}
)
handler.data_malloc()
x.copyin_float([1, 2, 3, 4, -1, -2, -3, -4])
w.copyin_float([1, 1, 1, 1])
handler.run()
print(y.copyout_numpy())
```

同一入口支持 Tensor、Tensor 列表、整数、浮点数、布尔、字符串、整数列表和相应
optional 参数。缺省 optional 参数传空值；没有公开缺省重载的必需参数必须显式提供。
DataType 属性可使用 ONNX 类型编号或 InfiniRT 类型名称（如 `float32`、`float64`）。
算子名称保留相对命名空间，例如 `special::Erf`、`linalg::Inv`。
生成清单会逐一记录发现的工厂，每个工厂均有已绑定签名或明确的不支持原因。
生成清单位于构建目录的 `generated/infiniops_bindings.json`，包含 SDK 摘要、签名及
暂不支持的参数类型；SDK 声明格式不兼容时构建会明确失败。

通用节点要求前端提供正确的输出形状和 dtype，不从 C++ 参数签名猜测算子数学语义。
动态改变输入形状时，先调用 `op.set_output_shapes([...])`，再调用
`handler.shape_infer()` 和 `handler.data_malloc()`。ONNX 中的新语义仍需要前端完成
属性解释与形状推导；语义和名称一致的既有简单算子可直接使用自动生成的绑定，
旧 IR 与 InfiniOps 参数有差异时只在 `lowering.cc` 保留语义适配。

普通执行和 Graph 共用同一份准备计划、实际算子实例和工作区。
准备时先计算临时张量布局，再分配一块按节点峰值复用的临时区；所有算子的工作区
共用另一块按最大请求分配的空间。只读常量独立保留，不会被后续节点的临时结果覆盖。
共享区域在同一 runtime stream 上按节点顺序使用，Graph 捕获期间地址不变。
`clear_graph_cache()` 会同步已提交工作并释放相应准备资源，包括普通执行缓存；
捕获失败和 LRU 淘汰也会解除匹配的普通计划引用。

InfiniOps 负责选择实现并声明捕获安全性；InfiniRT 负责图捕获、实例化及重放。
未声明安全的 provider 在捕获前被拒绝，仍可普通执行。
RMSNorm 在形状推导及调用前检查非标量输入、正的末维、权重长度、输出形状及 dtype；
固定 SDK 的执行接口仅支持 rank 2／3，其他秩在创建算子前明确拒绝。
旧 RoPE 节点需要根据运行时位置 ID 生成表，按拓扑顺序读取位置并执行，暂不支持捕获。
该后端使用 InfiniOps 的自动选择，不支持旧的 InfiniTensor kernel 调优入口；
`handler.tune()` 会明确报错，不影响 `handler.run()`。

回归命令：

```bash
python scripts/tests/test_generate_infiniops_bindings.py
# SDK 需包含 RmsNorm、Cat、Softmax、Abs、Erf；缺少能力直接失败。
python pyinfinitensor/tests/test_infiniops_bindings.py
```
