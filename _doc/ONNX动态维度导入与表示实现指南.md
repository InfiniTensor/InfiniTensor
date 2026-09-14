# ONNX 动态维度导入与表示实现指南

## 1. 本阶段的目标和边界

本阶段只完成以下任务：

> 完善 ONNX 动态维度的导入和表示，正确区分固定维度、符号动态维度、未命名未知维度，以及每次执行时的实际 Shape。

本阶段不实现：

- `Shape → Gather → Unsqueeze → Concat` Shape Tensor 子图执行；
- Tensor 驱动的动态 `Reshape`；
- CUDA 动态 Shape；
- 符号表达式计算，例如 `height / 2`、`sequence + 1`；
- 对输出和中间 Tensor 建立完整的符号 Shape 推导系统。

当前阶段应该保留 C++ `TensorObj` 的具体 Shape 设计，不要把字符串或 `-1` 写入 `TensorObj::shape`。

最终的数据模型应当是：

```text
ONNX 声明 Shape
["batch", 3, unknown, "width"]
        ↓ 保存在 OnnxStub
[Symbolic("batch"), Fixed(3), Unknown, Symbolic("width")]

首次构图使用的具体 Shape
[1, 3, 1, 1]
        ↓ 写入 TensorObj

某次运行的实际 Shape
[4, 3, 224, 320]
        ↓ 校验通过后写入 TensorObj
```

职责划分：

| 组件 | 负责的 Shape 信息 |
|---|---|
| `OnnxStub` | ONNX 声明约束：fixed、symbolic、unknown |
| `TensorObj` | 当前模型实例本次执行使用的具体整数 Shape |
| `Operator::inferShape()` | 根据具体输入 Shape 推导具体输出 Shape |
| `GraphObj::dataMalloc()` | 根据具体 Shape 计算出的字节数规划内存 |

## 2. 当前代码的问题

当前入口位于：

```text
pyinfinitensor/src/pyinfinitensor/onnx.py
```

当前实现：

```python
def _take_shape_dim(shape: TensorShapeProto) -> List[int]:
    return [(d.dim_value if d.dim_value > 0 else 1) for d in shape.dim]
```

它会产生不可逆的信息丢失：

| 原始 ONNX 维度 | 当前转换结果 |
|---|---:|
| 固定维 `dim_value = 1` | `1` |
| 符号维 `dim_param = "batch"` | `1` |
| 未指定维度 | `1` |
| 固定零维 `dim_value = 0` | `1`，这是错误的 |

转换后，`set_input()` 无法知道输入的某一维能不能变化，只能把用户传入的 Shape 直接写入 C++ Tensor。

需要把现在的单一转换拆成两个步骤：

```text
TensorShapeProto
    ├── parse → 声明 Shape 约束
    └── materialize → 首次构图所需的具体 Shape
```

## 3. 需要修改的文件

基础实现只需要修改：

```text
pyinfinitensor/src/pyinfinitensor/onnx.py
pyinfinitensor/tests/test_onnxstub.py
```

可选的标量输入支持需要额外修改：

```text
src/core/graph_handler.cc
```

当前阶段不应修改：

```text
include/core/tensor.h
src/core/tensor.cc
include/core/graph.h
src/core/graph.cc
src/core/lazy_allocator.cc
```

这些文件处理的是具体 Shape 和内存，现有能力可以直接复用。

## 4. 第一步：定义动态维度的数据结构

### 4.1 修改 imports

在 `onnx.py` 文件顶部增加：

```python
import operator
from dataclasses import dataclass
from enum import Enum
```

项目声明支持 Python 3.7，因此类型标注继续使用 `typing.List`、`typing.Tuple`，不要使用 Python 3.9 才完整支持的 `list[int]` 和 `tuple[int, ...]`。

### 4.2 增加维度类型

建议放在 `OnnxStub` 类之前：

```python
class DimensionKind(str, Enum):
    """The kind of one dimension in an ONNX tensor shape."""

    FIXED = "fixed"
    SYMBOLIC = "symbolic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DimensionSpec:
    """The shape constraint declared by one ONNX dimension."""

    kind: DimensionKind
    value: Optional[int] = None
    symbol: Optional[str] = None


ShapeSpec = Tuple[DimensionSpec, ...]
```

这里使用不可变 dataclass 和 tuple，避免调用者意外修改模型声明。

三个状态的含义：

```python
DimensionSpec(DimensionKind.FIXED, value=3)
DimensionSpec(DimensionKind.SYMBOLIC, symbol="batch")
DimensionSpec(DimensionKind.UNKNOWN)
```

不要使用以下有歧义的表示：

```python
-1                  # 无法区分 symbolic 和 unknown
None                # 无法携带 dim_param 名字
"batch"             # 无法统一表达 fixed/unknown
```

## 5. 第二步：实现无损解析和具体化

在 `onnx.py` 底部辅助函数区域增加以下函数。

### 5.1 解析 ONNX 声明

```python
def _parse_shape_spec(shape: TensorShapeProto) -> ShapeSpec:
    dimensions = []

    for dimension in shape.dim:
        field = dimension.WhichOneof("value")

        if field == "dim_value":
            dimensions.append(
                DimensionSpec(
                    kind=DimensionKind.FIXED,
                    value=int(dimension.dim_value),
                )
            )
        elif field == "dim_param" and dimension.dim_param:
            dimensions.append(
                DimensionSpec(
                    kind=DimensionKind.SYMBOLIC,
                    symbol=dimension.dim_param,
                )
            )
        else:
            # Neither dim_value nor a non-empty dim_param is available.
            dimensions.append(DimensionSpec(kind=DimensionKind.UNKNOWN))

    return tuple(dimensions)
```

注意事项：

1. 必须使用 `WhichOneof("value")`，不要继续使用 `dim_value > 0`；
2. `dim_value = 0` 是一个固定维度，不是动态维；
3. 空字符串 `dim_param` 按 unknown 处理；
4. 标量 Tensor 的 `shape.dim` 为空，返回空 tuple，这是合法情况。

### 5.2 生成首次构图 Shape

```python
def _materialize_shape(
    shape_spec: ShapeSpec,
    dynamic_default: int = 1,
) -> List[int]:
    if dynamic_default < 0:
        raise ValueError("dynamic_default must be non-negative")

    dimensions = []
    for dimension in shape_spec:
        if dimension.kind == DimensionKind.FIXED:
            if dimension.value is None:
                raise ValueError("A fixed dimension must have a value")
            dimensions.append(dimension.value)
        else:
            dimensions.append(dynamic_default)

    return dimensions
```

### 5.3 判断模型是否有动态输入

```python
def _shape_spec_is_dynamic(shape_spec: ShapeSpec) -> bool:
    return any(
        dimension.kind != DimensionKind.FIXED
        for dimension in shape_spec
    )
```

### 5.4 保留旧函数作为兼容包装

不要直接删除 `_take_shape_dim()`。仓库外部可能有人导入了这个辅助函数，保留一层包装可以减少无关破坏：

```python
def _take_shape_dim(shape: TensorShapeProto) -> List[int]:
    return _materialize_shape(_parse_shape_spec(shape))
```

## 6. 第三步：在 simplifier 之前保存原始输入约束

当前 `OnnxStub.__init__()` 会先运行 `onnxsim.simplify()`。动态模型存在以下风险：

```text
动态维以 1 作为初始值
        ↓
onnxsim 对 Shape 信息进行推导或常量折叠
        ↓
动态语义可能被固定为首次取值
```

因此应当先从原始模型提取输入约束，并对动态输入模型跳过 simplifier。

### 6.1 替换 `OnnxStub.__init__()` 开头的模型处理

当前结构大致为：

```python
model = copy.deepcopy(model)
try:
    model_simp, check = simplify(copy.deepcopy(model))
    ...
```

建议改为：

```python
model = copy.deepcopy(model)

original_initializer_names = {
    initializer.name for initializer in model.graph.initializer
}

original_input_shape_specs = {
    input_info.name: _parse_shape_spec(input_info.type.tensor_type.shape)
    for input_info in model.graph.input
    if input_info.name not in original_initializer_names
}

has_dynamic_input = any(
    _shape_spec_is_dynamic(shape_spec)
    for shape_spec in original_input_shape_specs.values()
)

# Keep dynamic input and Shape semantics unchanged. Static models retain the
# existing simplification path.
if not has_dynamic_input:
    try:
        # onnx simplifier performs inplace simplify
        model_simp, check = simplify(copy.deepcopy(model))
        if check:
            model = model_simp
    except ValidationError:
        pass
    except RuntimeError:
        pass
```

关键原则：

- `original_input_shape_specs` 必须从未经 simplifier 修改的模型提取；
- initializer 即使同时列在 `graph.input` 中，也不能作为用户运行时输入保存；
- 动态模型先完全跳过 simplify，保证正确性；
- 静态模型保持原来的 simplify 行为，降低回归风险。

以后如果要恢复动态模型 simplify，需要证明 simplify 前后的动态输入和 Shape 子图语义等价，不属于本阶段范围。

## 7. 第四步：让 `OnnxStub` 保存声明 Shape 和实际 Shape

### 7.1 增加成员

在 `OnnxStub.__init__()` 初始化 `self.inputs` 的附近增加：

```python
self.input_shape_specs: Dict[str, ShapeSpec] = dict(
    original_input_shape_specs
)
self.current_input_shapes: Dict[str, Tuple[int, ...]] = {}
```

成员含义：

```text
input_shape_specs
    模型原始声明，模型生命周期内不随运行改变

current_input_shapes
    最近一次成功设置并完成初始化的实际输入 Shape
```

### 7.2 修改 graph input 创建逻辑

当前代码：

```python
for input in model.graph.input:
    dims = _take_shape_dim(input.type.tensor_type.shape)
    if input.name not in tensors.keys():
        tensors[input.name] = self.handler.tensor(
            dims, input.type.tensor_type.elem_type
        )
        tensors[input.name].set_input()
```

建议改为：

```python
for input_info in model.graph.input:
    if input_info.name not in tensors:
        shape_spec = self.input_shape_specs.get(input_info.name)
        if shape_spec is None:
            # Fallback for a static model changed by simplification.
            shape_spec = _parse_shape_spec(input_info.type.tensor_type.shape)
            self.input_shape_specs[input_info.name] = shape_spec

        dims = _materialize_shape(shape_spec)
        tensors[input_info.name] = self.handler.tensor(
            dims,
            input_info.type.tensor_type.elem_type,
        )
        tensors[input_info.name].set_input()
```

不要为 initializer 重复创建 input spec。initializer 已经提前进入 `tensors`，所以会被 `if input_info.name not in tensors` 排除。

### 7.3 初始化当前实际 Shape

在 `self.inputs` 全部填充完成之后、第一次 `self.init()` 之前增加：

```python
self.current_input_shapes = {
    name: tuple(tensor.shape())
    for name, tensor in self.inputs.items()
}
```

例如原始声明：

```text
x: ["batch", 2]
```

导入完成后：

```python
self.input_shape_specs["x"]
# (
#     DimensionSpec(SYMBOLIC, symbol="batch"),
#     DimensionSpec(FIXED, value=2),
# )

self.current_input_shapes["x"]
# (1, 2)
```

## 8. 第五步：增加可查询接口

模型导入后应当能够回答“哪些维度是动态的”。在 `OnnxStub` 中增加：

```python
def get_input_shape_spec(self, name: str) -> ShapeSpec:
    if name not in self.input_shape_specs:
        raise KeyError('Unknown model input "{}"'.format(name))
    return self.input_shape_specs[name]
```

返回值由 frozen dataclass 和 tuple 组成，可以直接返回，不需要深拷贝。

不要改变现有 `getShape()` 的语义：

```python
stub.get_input_shape_spec("x")
# 返回 ONNX 声明约束

stub.getShape("x")
# 返回当前执行的具体 Shape
```

两个接口不能混为一个。

## 9. 第六步：实现完整的输入 Shape 校验

### 9.1 校验规则

`set_input()` 修改任何 C++ Tensor 之前，必须完成所有输入的校验。

至少覆盖：

1. 输入数量与模型运行时输入数量相等；
2. 每个输入的 rank 与 ONNX 声明一致；
3. 每个实际维度都是整数；
4. 不接受 bool 作为维度；
5. 每个实际维度非负；
6. fixed 维必须等于声明值；
7. symbolic 和 unknown 维允许在不同运行之间变化；
8. 同一次执行中，相同 `dim_param` 必须取相同值；
9. unknown 维之间互相独立，不要求相等。

### 9.2 增加私有校验方法

在 `OnnxStub` 中增加：

```python
def _validate_input_shapes(
    self,
    input_shapes: List[Sequence[int]],
) -> Dict[str, List[int]]:
    if len(input_shapes) != len(self.inputs):
        raise ValueError(
            "input_shapes must contain one shape per model input; "
            "expected {}, got {}".format(
                len(self.inputs),
                len(input_shapes),
            )
        )

    normalized_shapes: Dict[str, List[int]] = {}
    # symbol -> (value, input_name, dimension_index)
    symbol_values: Dict[str, Tuple[int, str, int]] = {}

    for input_name, proposed_shape in zip(self.inputs, input_shapes):
        shape_spec = self.input_shape_specs[input_name]

        if len(proposed_shape) != len(shape_spec):
            raise ValueError(
                'Input "{}" rank mismatch: expected {}, got {}'.format(
                    input_name,
                    len(shape_spec),
                    len(proposed_shape),
                )
            )

        normalized_shape = []
        for dimension_index, (raw_value, dimension_spec) in enumerate(
            zip(proposed_shape, shape_spec)
        ):
            if isinstance(raw_value, (bool, np.bool_)):
                raise TypeError(
                    'Input "{}" dimension {} must be an integer, got bool'.format(
                        input_name,
                        dimension_index,
                    )
                )

            try:
                value = operator.index(raw_value)
            except TypeError:
                raise TypeError(
                    'Input "{}" dimension {} must be an integer, got {!r}'.format(
                        input_name,
                        dimension_index,
                        raw_value,
                    )
                )

            if value < 0:
                raise ValueError(
                    'Input "{}" dimension {} must be non-negative, got {}'.format(
                        input_name,
                        dimension_index,
                        value,
                    )
                )

            if dimension_spec.kind == DimensionKind.FIXED:
                expected = dimension_spec.value
                if value != expected:
                    raise ValueError(
                        'Input "{}" dimension {} is fixed at {}, but received {}'.format(
                            input_name,
                            dimension_index,
                            expected,
                            value,
                        )
                    )

            elif dimension_spec.kind == DimensionKind.SYMBOLIC:
                symbol = dimension_spec.symbol
                if symbol is None:
                    raise ValueError(
                        "A symbolic dimension must have a non-empty name"
                    )

                previous = symbol_values.get(symbol)
                if previous is None:
                    symbol_values[symbol] = (
                        value,
                        input_name,
                        dimension_index,
                    )
                elif previous[0] != value:
                    raise ValueError(
                        'Symbolic dimension "{}" has conflicting values: '
                        '{}[{}]={}, {}[{}]={}'.format(
                            symbol,
                            previous[1],
                            previous[2],
                            previous[0],
                            input_name,
                            dimension_index,
                            value,
                        )
                    )

            # UNKNOWN accepts any non-negative integer and has no equality
            # relationship with another UNKNOWN dimension.
            normalized_shape.append(value)

        normalized_shapes[input_name] = normalized_shape

    return normalized_shapes
```

为什么使用 `operator.index()`：

- 接受 Python `int`；
- 接受 `np.int32`、`np.int64` 等整数对象；
- 拒绝浮点数和字符串；
- bool 需要在调用前单独拒绝，因为 Python bool 是 int 的子类。

### 9.3 修改 `set_input()`

当前方法会一边遍历一边修改 Tensor。如果第二个输入失败，第一个输入可能已经改变。

改成：

```python
def set_input(self, inputShapes: List[Sequence[int]]) -> None:
    normalized_shapes = self._validate_input_shapes(inputShapes)

    # No C++ Tensor is modified before all frontend constraints pass.
    for input_name in self.inputs:
        old_tensor = self.inputs[input_name]
        self.handler.change_shape(
            normalized_shapes[input_name],
            old_tensor.fuid(),
        )

    self.handler.shape_infer()
    self.init()

    self.current_input_shapes = {
        input_name: tuple(shape)
        for input_name, shape in normalized_shapes.items()
    }
```

保持参数名 `inputShapes` 可以减少现有调用者变化；错误信息内部可以统一使用 `input_shapes`。

这里实现的是“前端校验的原子性”：任何声明 Shape 校验失败都不会修改 C++ Tensor。

如果后续还需要保证 C++ `shape_infer()` 或内存分配失败时也能回滚，需要额外保存旧 Shape 并建立异常恢复流程。基础任务暂时不需要把这个事务扩大到内存分配层。

## 10. 可选改动：支持标量模型输入

ONNX 标量 Shape 是：

```python
[]
```

`_parse_shape_spec()` 和 `_materialize_shape()` 已经能正确得到空 tuple 和空 list，但当前 C++ 接口在 `src/core/graph_handler.cc` 中拒绝空 Shape：

```cpp
IT_ASSERT(shape.size() != 0);
```

如果本阶段要覆盖标量输入，可以删除这一行：

```cpp
void GraphHandlerObj::change_shape(const vector<int> &shape, int tensorId) {
    auto tensor = g->getTensor(tensorId);
    IT_ASSERT(tensor != nullptr);
    tensor->setShape(shape);
}
```

`TensorObj` 已将空 Shape 解释为包含一个元素的标量。删除断言后仍需增加一个 Identity 标量输入测试。

如果提交时间紧，可以只保证标量声明能够被解析，在使用说明中注明当前 `set_input([])` 的限制。

## 11. 测试修改方案

测试集中放在：

```text
pyinfinitensor/tests/test_onnxstub.py
```

测试可以通过 `onnx_frontend.DimensionKind`、`onnx_frontend.DimensionSpec` 和辅助函数访问新增定义，避免增加大量 import。

### 11.1 测试 fixed、symbolic、unknown 和固定零维

```python
def test_parse_input_shape_spec(self):
    info = value_info("x", ["batch", 3, None, 0])

    spec = onnx_frontend._parse_shape_spec(
        info.type.tensor_type.shape
    )

    self.assertEqual(
        spec,
        (
            onnx_frontend.DimensionSpec(
                onnx_frontend.DimensionKind.SYMBOLIC,
                symbol="batch",
            ),
            onnx_frontend.DimensionSpec(
                onnx_frontend.DimensionKind.FIXED,
                value=3,
            ),
            onnx_frontend.DimensionSpec(
                onnx_frontend.DimensionKind.UNKNOWN,
            ),
            onnx_frontend.DimensionSpec(
                onnx_frontend.DimensionKind.FIXED,
                value=0,
            ),
        ),
    )
    self.assertEqual(
        onnx_frontend._materialize_shape(spec),
        [1, 3, 1, 0],
    )
```

### 11.2 测试导入后同时保留声明和实际 Shape

```python
def test_dynamic_input_keeps_declared_and_concrete_shapes(self):
    model = make_model(
        [helper.make_node("Identity", ["x"], ["y"])],
        [value_info("x", ["batch", 2])],
        [value_info("y", ["batch", 2])],
    )

    stub = import_model(model)

    self.assertEqual(
        stub.get_input_shape_spec("x"),
        (
            onnx_frontend.DimensionSpec(
                onnx_frontend.DimensionKind.SYMBOLIC,
                symbol="batch",
            ),
            onnx_frontend.DimensionSpec(
                onnx_frontend.DimensionKind.FIXED,
                value=2,
            ),
        ),
    )
    self.assertEqual(stub.getShape("x"), [1, 2])
    self.assertEqual(stub.current_input_shapes["x"], (1, 2))
```

### 11.3 测试动态维连续变化

```python
def test_symbolic_dimension_changes_between_runs(self):
    model = make_model(
        [helper.make_node("Identity", ["x"], ["y"])],
        [value_info("x", ["batch", 2])],
        [value_info("y", ["batch", 2])],
    )
    stub = import_model(model)

    for batch in (1, 2, 8, 3, 1):
        stub.set_input([[batch, 2]])
        self.assertEqual(stub.getShape("x"), [batch, 2])
        self.assertEqual(stub.getShape("y"), [batch, 2])
        self.assertEqual(stub.current_input_shapes["x"], (batch, 2))
```

### 11.4 测试固定维变化被前端拒绝

```python
def test_fixed_dimension_change_is_rejected_without_mutation(self):
    model = make_model(
        [helper.make_node("Identity", ["x"], ["y"])],
        [value_info("x", ["batch", 2])],
        [value_info("y", ["batch", 2])],
    )
    stub = import_model(model)
    before = stub.getShape("x")

    with self.assertRaisesRegex(
        ValueError,
        r'dimension 1 is fixed at 2, but received 5',
    ):
        stub.set_input([[4, 5]])

    self.assertEqual(stub.getShape("x"), before)
    self.assertEqual(stub.getShape("y"), before)
```

### 11.5 测试 rank 和维度类型

```python
def test_invalid_concrete_input_shapes_are_rejected(self):
    model = make_model(
        [helper.make_node("Identity", ["x"], ["y"])],
        [value_info("x", ["batch", 2])],
        [value_info("y", ["batch", 2])],
    )
    stub = import_model(model)

    invalid_cases = (
        ([[4, 2, 1]], ValueError, "rank mismatch"),
        ([[-1, 2]], ValueError, "must be non-negative"),
        ([[1.5, 2]], TypeError, "must be an integer"),
        ([[True, 2]], TypeError, "got bool"),
    )

    for shapes, error_type, message in invalid_cases:
        with self.subTest(shapes=shapes):
            with self.assertRaisesRegex(error_type, message):
                stub.set_input(shapes)
```

### 11.6 测试同名符号一致性和多输入原子校验

```python
def test_shared_symbol_must_match_without_partial_update(self):
    model = make_model(
        [helper.make_node("Add", ["x", "z"], ["y"])],
        [
            value_info("x", ["batch", 2]),
            value_info("z", ["batch", 2]),
        ],
        [value_info("y", ["batch", 2])],
    )
    stub = import_model(model)

    x_before = stub.getShape("x")
    z_before = stub.getShape("z")

    with self.assertRaisesRegex(
        ValueError,
        'Symbolic dimension "batch" has conflicting values',
    ):
        stub.set_input([[4, 2], [3, 2]])

    self.assertEqual(stub.getShape("x"), x_before)
    self.assertEqual(stub.getShape("z"), z_before)
```

### 11.7 测试 unknown 维之间互相独立

```python
def test_unknown_dimensions_are_independent(self):
    model = make_model(
        [
            helper.make_node("Identity", ["x"], ["x_out"]),
            helper.make_node("Identity", ["z"], ["z_out"]),
        ],
        [
            value_info("x", [None, 2]),
            value_info("z", [None, 2]),
        ],
        [
            value_info("x_out", [None, 2]),
            value_info("z_out", [None, 2]),
        ],
    )
    stub = import_model(model)

    stub.set_input([[4, 2], [3, 2]])

    self.assertEqual(stub.getShape("x"), [4, 2])
    self.assertEqual(stub.getShape("z"), [3, 2])
```

### 11.8 测试动态模型跳过 simplifier

这个测试不要使用现有 `import_model()` helper，因为该 helper 自己会 patch `simplify`。直接构造 `OnnxStub`：

```python
def test_dynamic_input_skips_simplifier(self):
    model = make_model(
        [helper.make_node("Identity", ["x"], ["y"])],
        [value_info("x", ["batch", 2])],
        [value_info("y", ["batch", 2])],
    )

    with patch.object(onnx_frontend, "simplify") as simplify_mock:
        OnnxStub(model, backend.cpu_runtime())

    simplify_mock.assert_not_called()
```

还应保留现有静态模型测试，确认静态导入行为没有变化。

## 12. 修改完成后的完整代码流程

### 12.1 模型导入

以输入 `x: ["batch", 2]` 为例：

```text
原始 ModelProto
        ↓
_parse_shape_spec()
        ↓
(Symbolic("batch"), Fixed(2))
        ↓ 保存到 input_shape_specs
_materialize_shape()
        ↓
[1, 2]
        ↓ GraphHandler.tensor()
TensorObj(shape=[1, 2])
        ↓
current_input_shapes["x"] = (1, 2)
```

### 12.2 合法运行时 Shape

```text
set_input([[4, 2]])
        ↓
检查输入数量
        ↓
检查 rank = 2
        ↓
第 0 维 symbolic("batch")：接受 4
第 1 维 fixed(2)：收到 2，合法
        ↓ 全部输入校验通过
change_shape([4, 2])
        ↓
graph.shape_infer()
        ↓
dataMalloc()
        ↓
current_input_shapes["x"] = (4, 2)
```

### 12.3 非法固定维变化

```text
set_input([[4, 5]])
        ↓
第 1 维 fixed(2)，但收到 5
        ↓
抛出 ValueError
        ↓
不调用 change_shape()
        ↓
X、中间 Tensor、内存状态均不改变
```

### 12.4 同名符号冲突

```text
x:    ["batch", 2] ← 实际 [4, 2]
mask: ["batch", 2] ← 实际 [3, 2]
        ↓
symbol_values["batch"] 第一次记录 4
        ↓
第二次得到 3，与 4 冲突
        ↓
在修改任何 Tensor 前拒绝
```

## 13. 建议的开发和验证顺序

不要一次写完所有代码再运行测试。按以下顺序推进：

1. 增加 `DimensionKind`、`DimensionSpec`；
2. 实现 `_parse_shape_spec()`；
3. 实现 `_materialize_shape()`；
4. 只运行解析函数测试；
5. 在 simplifier 前提取原始 input specs；
6. 接入 input Tensor 创建流程；
7. 增加 `get_input_shape_spec()`；
8. 实现 `_validate_input_shapes()`；
9. 改造 `set_input()` 为先校验、后修改；
10. 增加连续动态 Shape 和错误输入测试；
11. 运行全部 Python ONNX 测试；
12. 运行 C++ 回归测试，确认没有间接回归。

## 14. 构建和测试命令

仓库主要面向 Linux/WSL。CPU Debug 构建：

```bash
make TYPE=Debug CUDA=OFF TEST=ON
make install-python TYPE=Debug CUDA=OFF TEST=ON
```

先运行直接相关测试：

```bash
python pyinfinitensor/tests/test_onnxstub.py
python pyinfinitensor/tests/test_onnx.py
```

再运行完整 CPU C++ 测试：

```bash
make test-cpp TYPE=Debug CUDA=OFF TEST=ON
```

最后运行项目定义的 ONNX 测试入口：

```bash
make test-onnx TYPE=Debug CUDA=OFF TEST=ON
```

格式检查：

```bash
python scripts/format.py
```

## 15. 完成判定清单

代码完成后逐项确认：

- [ ] `dim_value` 被识别为 fixed；
- [ ] `dim_param` 被识别为 symbolic，并保留名字；
- [ ] 未设置维度被识别为 unknown；
- [ ] 固定零维没有被替换成 `1`；
- [ ] 动态维仅在首次构图时具体化为 `1`；
- [ ] 模型导入后可以查询原始 input shape spec；
- [ ] `TensorObj` 中始终只有本次具体 Shape；
- [ ] fixed 维变化在 Python 前端被明确拒绝；
- [ ] symbolic/unknown 维可以连续变化；
- [ ] 同一次执行中相同 symbolic 名字取值一致；
- [ ] 多输入校验失败不会修改任何输入 Tensor；
- [ ] 动态模型不会经过可能固化 Shape 的 simplifier；
- [ ] 静态 ONNX 模型原有导入测试保持通过；
- [ ] 同一个模型实例可以完成 `1 → 2 → 8 → 3 → 1`；
- [ ] 新错误信息包含输入名、维度位置、期望值和实际值。

## 16. 本阶段建议的提交范围

建议形成一个独立提交：

```text
feat(onnx): preserve and validate dynamic input dimensions
```

提交只包含：

```text
pyinfinitensor/src/pyinfinitensor/onnx.py
pyinfinitensor/tests/test_onnxstub.py
```

如果实现标量输入，再包含：

```text
src/core/graph_handler.cc
```

不要在这个提交中混入 Shape Tensor Kernel、动态 Reshape 或 CUDA 改动。完成并验证这个阶段后，再开始下一条独立链路：运行时 Shape Tensor 子图执行。
