# 基础 Shape Tensor 相关算子实现指南

## 1. 本阶段的目标和边界

本阶段只完成训练营任务中的：

> 支持项目所需的基础 Shape Tensor 相关算子。

先跑通下面这条 Shape Tensor 数据链：

```text
X: [batch, 2, 3]
 │
 └─ Shape
      data = [batch, 2, 3], dtype = Int64
           │
           └─ Gather(index=0)
                data = batch
                     │
                     └─ Unsqueeze(axes=[0])
                          data = [batch]
                               │
Constant Int32 [6] ── Cast ────┤
                               │
                             Concat(axis=0)
                               data = [batch, 6]
```

验收结果示例：

```text
输入 X 的实际 Shape = [4, 2, 3]
Shape(X)             = [4, 2, 3]
Gather(..., 0)       = 4
Unsqueeze(4)         = [4]
Cast([6])            = Int64 [6]
Concat               = Int64 [4, 6]
```

这一阶段暂不实现：

- Shape Tensor 驱动的动态 `Reshape`；
- Runtime 两阶段执行；
- 动态 `Expand`；
- `ConstantOfShape`；
- CUDA/Ascend/Kunlun 等后端；
- `Shape` 的 `start/end` 切片；
- 运行时提供的 `Unsqueeze/Squeeze axes`。

第一版支持范围：

- 后端为 `Native CPU`；
- `Shape` 输出完整输入 Shape；
- `Gather` data 支持任意可按字节复制的 dtype，indices 支持 `Int32/Int64`；
- `Unsqueeze/Squeeze` 的 axes 在导入阶段是常量；
- `Concat` 至少支持 `Int32/Int64/Float32`；
- `Cast` 至少支持 Shape 子图常见的 `Int32 ↔ Int64`。

## 2. 先理解 Shape Tensor 是什么

必须区分下面两种信息。

### 2.1 Tensor 的 Shape 元数据

```cpp
x->getDims(); // [4, 2, 3]
```

它描述 Tensor `x` 的布局，不是普通 Tensor 数据。

### 2.2 Shape Tensor 的数据

```text
Tensor s = Shape(x)

s.getDims() = [3]       // s 自身是长度为 3 的一维 Tensor
s.getDType() = Int64
s.data = [4, 2, 3]      // 普通 Tensor 数据
```

`Gather/Unsqueeze/Concat/Cast` 处理的是 `s.data`。后续动态 `Reshape` 也会读取 Shape Tensor 的数据，而不是读取 `s.getDims()`。

## 3. 当前仓库能力盘点

| 算子 | Operator | ONNX importer | Native CPU Kernel | 当前缺口 |
|---|---:|---:|---:|---|
| `Shape` | 已有 | 已有 | 没有 | 输出 dtype 错误；没有 workload/attr；没有数值 Kernel |
| `Gather` | 已有 | 已有 | 没有 | Native CPU Kernel 缺失；负 index 检查不符合 ONNX |
| `Unsqueeze` | 已有 | 已有 | 已有 | Kernel 可直接复制 Int64；建议修复 inferShape 的成员副作用 |
| `Squeeze` | 已有 | 已有 | 已有 | Kernel 可直接复制 Int64；建议修复 inferShape 的成员副作用 |
| `Concat` | 已有 | 已有 | 已有 | CPU Kernel 未分派 Int32/Int64 |
| `Cast` | 已有 | 已有 | 没有 | Native CPU Kernel 缺失；性能键未包含 cast 类型 |
| `Constant` | 作为 weight Tensor 导入 | 已有 | 不需要 | initializer/Constant 的数据由 `OnnxStub.init()` 复制 |

需要修改或新增：

```text
include/operators/unary.h
src/operators/unary.cc
src/kernels/cpu/shape.cc                 新增

include/operators/gather.h
src/operators/gather.cc
src/kernels/cpu/gather.cc                新增

src/operators/unsqueeze.cc
src/operators/squeeze.cc

src/operators/concat.cc
src/kernels/cpu/concat.cc

src/operators/unary.cc
src/kernels/cpu/cast.cc                  新增

pyinfinitensor/src/pyinfinitensor/onnx.py

test/operators/test_shape.cc             新增
test/kernels/nativecpu/test_nativecpu_shape_tensor.cc  新增
pyinfinitensor/tests/test_onnxstub.py
```

项目的 `CMakeLists.txt` 使用 glob 收集：

```text
src/kernels/cpu/*.cc
test/operators/*.cc
test/kernels/nativecpu/*.cc
```

因此按上述目录新增文件后不需要手动列入源码表，但应重新运行一次 CMake configure。

## 4. 第一步：实现 `Shape`

### 4.1 修改 `include/operators/unary.h`

当前 `ShapeObj` 只有 Shape 推导。将它改为：

```cpp
class ShapeObj : public OperatorObj {
  public:
    ShapeObj(GraphObj *graph, Tensor input, Tensor output);
    OP_CLONE(ShapeObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
```

为什么要增加 workload/attr：`CpuRuntimeObj::run()` 即使不 tune，也会构造 `OpPerfKey`。若 `ShapeObj` 没有覆盖 `getWorkloadVector()`，会进入 `OperatorObj` 默认的 `IT_TODO_HALT()`，导致 Kernel 还没执行就停止。

### 4.2 修改 `src/operators/unary.cc`

保留现有构造函数和 `inferShape()`：

```cpp
ShapeObj::ShapeObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Shape, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ShapeObj::inferShape(const TensorVec &inputs) {
    return {{{static_cast<int>(inputs[0]->getRank())}}};
}
```

补充：

```cpp
vector<DataType> ShapeObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 1);
    return {DataType::Int64};
}

vector<int> ShapeObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape inputShape = inputs[0]->getDims();
    ret.insert(ret.end(), inputShape.begin(), inputShape.end());
    return ret;
}

vector<int> ShapeObj::getOpAttrVector() const {
    return {type.underlying()};
}
```

ONNX `Shape` 的输出始终是 `Int64`，不能继承输入 dtype。

### 4.3 新增 `src/kernels/cpu/shape.cc`

```cpp
#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {

class ShapeCpu final : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ShapeObj>(_op);
        IT_ASSERT(op != nullptr);
        IT_ASSERT(op->getOutput()->getDType() == DataType::Int64);

        const Shape inputShape = op->getInputs(0)->getDims();
        auto *output = op->getOutput()->getRawDataPtr<int64_t *>();

        for (size_t i = 0; i < inputShape.size(); ++i)
            output[i] = static_cast<int64_t>(inputShape[i]);
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Shape, ShapeCpu, "Shape_CPU");

} // namespace infini
```

这里不要读取输入 Blob。`Shape` 读取的是输入 Tensor 当前的 Shape 元数据：

```cpp
op->getInputs(0)->getDims()
```

## 5. 第二步：实现 Native CPU `Gather`

### 5.1 修改 Operator 层

文件：`include/operators/gather.h`

这里要区分“检查语义”和“检查位置”：

- Gather 的 indices 越界检查必须保留，不能丢掉。
- `inferShape()` 只处理 Tensor 的 Shape 元数据，不应读取 indices 的运行时数值。
- `// TODO: should check everytime index updated.` 的含义正是：现有检查放在 Shape inference 阶段只会偶然执行，无法保证 indices 每次更新后都重新检查。
- 因此，应把等价的越界检查迁移到 Gather Kernel 的 `compute()` 路径，使它在每次真正使用 indices 前执行。

修改顺序建议如下：

1. 先从 `inferShape()` 中删除 `IT_ASSERT(CheckIndexValid());`。
2. 在 Native CPU Gather Kernel 中实现每次 `compute()` 时的检查，同时处理 ONNX 允许的负 index。
3. Kernel 检查完成后，再从 `GatherObj` 中删除旧的成员声明：

```cpp
bool CheckIndexValid() const;
```

文件：`src/operators/gather.cc`

从 `inferShape()` 中删除：

```cpp
IT_ASSERT(CheckIndexValid());
```

在 Kernel 已承担运行时检查后，再删除整个旧 `CheckIndexValid()` 实现。不要只注释头文件中的声明而保留 `.cc` 中的定义，否则 C++ 会报“没有与该定义匹配的成员声明”。

旧实现不适合直接保留在 Operator 层的另一个原因是：它通过 `copyBlobToCPU()` 把任意设备上的 indices 拷回 CPU，使纯元数据的 Operator 验证与 Runtime/设备数据发生耦合。Native CPU Kernel 可以直接检查 CPU 内存中的 indices；其他后端则应在各自的 Kernel 或底层库调用路径中保证边界安全。

最终 Shape/dtype 推导保持为：

```cpp
optional<vector<Shape>> GatherObj::inferShape(const TensorVec &inputs) {
    const Shape dataShape = inputs[0]->getDims();
    const Shape indicesShape = inputs[1]->getDims();

    Shape outputShape = dataShape;
    outputShape.erase(outputShape.begin() + axis);
    outputShape.insert(outputShape.begin() + axis,
                       indicesShape.begin(), indicesShape.end());
    return {{outputShape}};
}

vector<DataType> GatherObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2);
    const auto indicesType = inputs[1]->getDType();
    IT_ASSERT(indicesType == DataType::Int32 ||
                  indicesType == DataType::Int64,
              "Gather indices must be Int32 or Int64");
    return {inputs[0]->getDType()};
}
```

ONNX Gather 输出 Shape 规则：

```text
data.shape[:axis]
+ indices.shape
+ data.shape[axis + 1:]
```

例如：

```text
data.shape    = [3]
indices.shape = []     // scalar index
axis          = 0
output.shape  = []     // scalar
```

### 5.2 新增 `src/kernels/cpu/gather.cc`

下面实现支持任意 data dtype，因为数据按元素字节块复制；只需要分派 indices dtype。

```cpp
#include "core/kernel.h"
#include "operators/gather.h"
#include <cstring>

namespace infini {

class GatherCpu final : public CpuKernelWithoutConfig {
    template <typename IndexT>
    void doCompute(const Operator &_op) const {
        auto op = as<GatherObj>(_op);
        IT_ASSERT(op != nullptr);

        const Tensor data = op->getInputs(0);
        const Tensor indices = op->getInputs(1);
        const Tensor output = op->getOutput();
        const Shape dataShape = data->getDims();
        const int axis = op->getAxis();

        size_t outer = 1;
        for (int i = 0; i < axis; ++i)
            outer *= static_cast<size_t>(dataShape[i]);

        size_t inner = 1;
        for (size_t i = static_cast<size_t>(axis + 1);
             i < dataShape.size(); ++i)
            inner *= static_cast<size_t>(dataShape[i]);

        const int64_t axisDim = dataShape[axis];
        const size_t indexCount = indices->size();
        const size_t elementBytes = data->getDType().getSize();
        const size_t copyBytes = inner * elementBytes;

        const auto *indexData =
            indices->getRawDataPtr<const IndexT *>();
        const auto *input = data->getRawDataPtr<const uint8_t *>();
        auto *result = output->getRawDataPtr<uint8_t *>();

        for (size_t outerIndex = 0; outerIndex < outer; ++outerIndex) {
            for (size_t indexOffset = 0;
                 indexOffset < indexCount; ++indexOffset) {
                int64_t index = static_cast<int64_t>(indexData[indexOffset]);

                // ONNX Gather 允许负 index。
                IT_ASSERT(index >= -axisDim && index < axisDim,
                          "Gather index is out of range");
                if (index < 0)
                    index += axisDim;

                const size_t inputElementOffset =
                    (outerIndex * static_cast<size_t>(axisDim) +
                     static_cast<size_t>(index)) *
                    inner;
                const size_t outputElementOffset =
                    (outerIndex * indexCount + indexOffset) * inner;

                std::memcpy(result + outputElementOffset * elementBytes,
                            input + inputElementOffset * elementBytes,
                            copyBytes);
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        const DataType indicesType = _op->getInputs(1)->getDType();

        if (indicesType == DataType::Int32)
            doCompute<int32_t>(_op);
        else if (indicesType == DataType::Int64)
            doCompute<int64_t>(_op);
        else
            IT_ASSERT(false, "Gather indices must be Int32 or Int64");
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Gather, GatherCpu, "Gather_CPU");

} // namespace infini
```

## 6. 第三步：确认 `Unsqueeze/Squeeze` 可处理 Shape Tensor

Native CPU 已在 `src/kernels/cpu/reshape.cc` 中注册：

```cpp
REGISTER_KERNEL(Device::CPU, OpType::Unsqueeze, NaiveIdentity, ...);
REGISTER_KERNEL(Device::CPU, OpType::Squeeze, NaiveIdentity, ...);
```

它们只改变 Tensor Shape，不改变元素顺序，所以对 Int64 数据执行 `memcpy` 是正确的，无需新增 Kernel。

但是现有 `inferShape()` 会修改成员 `axes`，反复调用 `graph->shape_infer()` 时可能产生副作用，建议修复。

### 6.1 修改 `src/operators/unsqueeze.cc`

将 `inferShape()` 改为只修改局部变量：

```cpp
optional<vector<Shape>> UnsqueezeObj::inferShape(const TensorVec &inputs) {
    const Shape inputShape = inputs[0]->getDims();
    const int outputRank =
        static_cast<int>(inputs[0]->getRank() + axes.size());

    Shape normalizedAxes = axes;
    Shape outputShape(outputRank, -1);

    for (auto &axis : normalizedAxes) {
        axis = get_real_axis(axis, outputRank);
        IT_ASSERT(outputShape[axis] == -1, "Unsqueeze axes have duplicate");
        outputShape[axis] = 1;
    }

    auto inputIt = inputShape.begin();
    for (auto &dimension : outputShape) {
        if (dimension == -1)
            dimension = *inputIt++;
    }
    return {{outputShape}};
}
```

### 6.2 修改 `src/operators/squeeze.cc`

文件顶部确保有：

```cpp
#include <algorithm>
```

将 `inferShape()` 改为：

```cpp
optional<vector<Shape>> SqueezeObj::inferShape(const TensorVec &inputs) {
    const Shape inputShape = inputs[0]->getDims();
    const int rank = inputs[0]->getRank();
    Shape normalizedAxes = axes;

    if (normalizedAxes.empty()) {
        for (int i = 0; i < rank; ++i) {
            if (inputShape[i] == 1)
                normalizedAxes.emplace_back(i);
        }
    }

    for (auto &axis : normalizedAxes) {
        axis = get_real_axis(axis, rank);
        IT_ASSERT(inputShape[axis] == 1,
                  "Squeeze axis must refer to a dimension of size 1");
    }

    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    IT_ASSERT(std::adjacent_find(normalizedAxes.begin(),
                                 normalizedAxes.end()) ==
                  normalizedAxes.end(),
              "Squeeze axes have duplicate");

    Shape outputShape;
    for (int i = 0; i < rank; ++i) {
        if (!std::binary_search(normalizedAxes.begin(),
                                normalizedAxes.end(), i))
            outputShape.emplace_back(inputShape[i]);
    }
    return {{outputShape}};
}
```

第一版 axes 仍由 C++ 属性保存。ONNX importer 已要求 axes 是 initializer/Constant；运行时 axes Tensor 留到后续扩展。

## 7. 第四步：让 `Concat` 支持 Int64

### 7.1 修改 `src/operators/concat.cc`

构造函数开头还应避免空 inputs：

```cpp
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    IT_ASSERT(!inputs.empty(), "Concat requires at least one input");
    const int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}
```

在 `inferShape()` 开头加入 dtype 一致性检查：

```cpp
IT_ASSERT(!inputs.empty(), "Concat requires at least one input");
const DataType dtype = inputs[0]->getDType();
for (const auto &input : inputs) {
    IT_ASSERT(input->getDType() == dtype,
              "Concat inputs must have the same dtype");
}
```

原有 Shape 推导可以保留。

### 7.2 修改 `src/kernels/cpu/concat.cc`

将 `NaiveConcat::compute()` 中的 dtype switch 改成：

```cpp
const int dataTypeIdx = _op->getDType().getIndex();
switch (dataTypeIdx) {
    CASE(1);  // Float32
    break;
    CASE(6);  // Int32
    break;
    CASE(7);  // Int64
    break;
    CASE(12); // UInt32
    break;
default:
    IT_TODO_HALT();
}
```

Shape Tensor 主要依赖 `CASE(7)`。

## 8. 第五步：实现 Native CPU `Cast`

### 8.1 修正 `CastObj` 性能键

文件：`src/operators/unary.cc`

当前 workload/attr 没有包含 `castType`，不同转换可能生成相同性能键。改为：

```cpp
vector<int> CastObj::getWorkloadVector() const {
    vector<int> ret{
        type.underlying(),
        static_cast<int>(castType),
    };
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> CastObj::getOpAttrVector() const {
    return {
        type.underlying(),
        static_cast<int>(castType),
    };
}
```

### 8.2 新增 `src/kernels/cpu/cast.cc`

第一版实现 Shape 子图常见类型：

```cpp
#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {

class CastCpu final : public CpuKernelWithoutConfig {
    template <typename InputT, typename OutputT>
    void doCompute(const Operator &_op) const {
        auto op = as<CastObj>(_op);
        IT_ASSERT(op != nullptr);

        const auto *input =
            op->getInputs(0)->getRawDataPtr<const InputT *>();
        auto *output = op->getOutput()->getRawDataPtr<OutputT *>();

        for (size_t i = 0; i < op->getOutput()->size(); ++i)
            output[i] = static_cast<OutputT>(input[i]);
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<CastObj>(_op);
        IT_ASSERT(op != nullptr);

        switch (op->getType()) {
        case CastType::Int322Int64:
            doCompute<int32_t, int64_t>(_op);
            break;
        case CastType::Int642Int32:
            doCompute<int64_t, int32_t>(_op);
            break;
        case CastType::Float2Int32:
            doCompute<float, int32_t>(_op);
            break;
        case CastType::Float2Int64:
            doCompute<float, int64_t>(_op);
            break;
        case CastType::Int322Float:
            doCompute<int32_t, float>(_op);
            break;
        case CastType::Int642Float:
            doCompute<int64_t, float>(_op);
            break;
        case CastType::Float2Float:
            doCompute<float, float>(_op);
            break;
        default:
            IT_ASSERT(false,
                      "Cast type is not supported by Native CPU");
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Cast, CastCpu, "Cast_CPU");

} // namespace infini
```

先只实现真实测试模型使用的转换。不要为了“看起来完整”同时实现 Float16/BFloat16；这些类型还涉及项目已有的半精度表示和转换工具，应该另行验证。

## 9. 第六步：检查 ONNX importer

文件：`pyinfinitensor/src/pyinfinitensor/onnx.py`

现有 importer 已有：

```text
Shape
Gather
Unsqueeze
Squeeze
Concat
Cast
Constant
```

大部分不需要重写，只需要明确暂不支持的语义，避免静默产生错误结果。

### 9.1 `Shape`

当前 `Shape` 分支会忽略 `start/end`。第一版建议改为：

```python
elif node.op_type == "Shape":
    attributes = _parse_attribute(
        node,
        {
            "start": 0,
            "end": None,
        },
    )

    if attributes["start"] != 0 or attributes["end"] is not None:
        raise NotImplementedError(
            "Shape start/end slicing is not supported yet"
        )

    tensors[node.output[0]] = self.handler.shape(
        tensors[node.input[0]],
        tensors.get(node.output[0]),
    )
```

### 9.2 `Gather`

当前分支可以保留：

```python
elif node.op_type == "Gather":
    tensors[node.output[0]] = self.handler.gather(
        tensors[node.input[0]],
        tensors[node.input[1]],
        tensors.get(node.output[0]),
        next(
            (attr.i for attr in node.attribute if attr.name == "axis"),
            0,
        ),
    )
```

### 9.3 `Unsqueeze/Squeeze`

当前 importer 将 axes 解析为静态列表再传给 GraphHandler。这足够支持项目最小链路：

```text
axes 是 initializer/Constant -> 支持
axes 是运行时计算 Tensor     -> 第一版明确不支持
```

不要把 axes initializer 同时作为第二个 C++ 输入接入现有 `UnsqueezeObj`，因为当前类的接口仍然只有一个 data Tensor 输入和一个 `Shape axes` 属性。

### 9.4 `Concat`

当前分支可以保留。Native CPU 加上 Int64 分派后即可执行 Shape Tensor 拼接。

### 9.5 `Cast`

当前分支可以保留。`GraphHandlerObj::inferCastType()` 已能把 ONNX dtype 编号映射到 `CastType`；Native CPU Kernel 只需覆盖实际使用的转换。

## 10. 测试一：Operator 元数据测试

新增 `test/operators/test_shape.cc`：

```cpp
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {

TEST(Shape, InferShapeAndDataType) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({4, 2, 3}, DataType::Float32);
    auto op = g->addOp<ShapeObj>(input, nullptr);

    EXPECT_EQ(op->getOutput()->getDims(), (Shape{3}));
    EXPECT_EQ(op->getOutput()->getDType(), DataType::Int64);
}

} // namespace infini
```

这个测试只验证：

```text
Shape 输出 Tensor 的元数据 Shape = [input rank]
Shape 输出 dtype = Int64
```

不验证 Kernel 数值。

## 11. 测试二：基础 Shape Tensor 完整链路

新增 `test/kernels/nativecpu/test_nativecpu_shape_tensor.cc`：

```cpp
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/gather.h"
#include "operators/squeeze.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include "test.h"

namespace infini {

TEST(NativeCpuShapeTensor, BasicChain) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // X 的数据不重要，Shape 只读取 X 的元数据。
    auto x = g->addTensor({4, 2, 3}, DataType::Float32);
    x->setInput();

    auto shape = g->addOp<ShapeObj>(x, nullptr)->getOutput();

    // scalar index 0；Gather([4,2,3], 0) 得到 scalar 4。
    auto index = g->addTensor({}, DataType::Int64);
    index->setWeight();
    auto batch =
        g->addOp<GatherObj>(shape, index, nullptr, 0)->getOutput();

    // scalar 4 -> 一维 [4]。
    auto batch1d =
        g->addOp<UnsqueezeObj>(batch, nullptr, Shape{0})->getOutput();

    // 用 Int32 常量验证 Cast。
    auto six32 = g->addTensor({1}, DataType::Int32);
    six32->setWeight();
    auto six64 =
        g->addOp<CastObj>(six32, nullptr,
                          CastType::Int322Int64)->getOutput();

    auto target =
        g->addOp<ConcatObj>(TensorVec{batch1d, six64}, nullptr, 0)
            ->getOutput();
    target->setOutput();

    // 再经过 Squeeze，单独验证 Int64 字节复制路径。
    auto batchAgain =
        g->addOp<SqueezeObj>(batch1d, nullptr, Shape{0})->getOutput();

    g->dataMalloc();
    index->copyin<int64_t>({0});
    six32->copyin<int32_t>({6});

    runtime->run(g);

    EXPECT_TRUE(shape->equalData<int64_t>({4, 2, 3}));
    EXPECT_EQ(batch->getDims(), (Shape{}));
    EXPECT_TRUE(batch->equalData<int64_t>({4}));
    EXPECT_EQ(batch1d->getDims(), (Shape{1}));
    EXPECT_TRUE(batch1d->equalData<int64_t>({4}));
    EXPECT_TRUE(six64->equalData<int64_t>({6}));
    EXPECT_TRUE(target->equalData<int64_t>({4, 6}));
    EXPECT_TRUE(batchAgain->equalData<int64_t>({4}));
}

TEST(NativeCpuShapeTensor, GatherNegativeIndex) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto data = g->addTensor({3}, DataType::Int64);
    auto index = g->addTensor({}, DataType::Int64);
    data->setInput();
    index->setInput();

    auto output =
        g->addOp<GatherObj>(data, index, nullptr, 0)->getOutput();
    output->setOutput();

    g->dataMalloc();
    data->copyin<int64_t>({4, 2, 3});
    index->copyin<int64_t>({-1});
    runtime->run(g);

    EXPECT_TRUE(output->equalData<int64_t>({3}));
}

} // namespace infini
```

注意：上面同一个 `batch1d` 同时被 `Concat` 和 `Squeeze` 使用，现有图内存规划会根据 target 引用数处理它，不需要复制输入。

如果你希望失败定位更简单，也可以先把每个算子拆成单独 TEST，全部通过后再保留一条完整链路测试。

## 12. 测试三：ONNX 导入和执行

在 `pyinfinitensor/tests/test_onnxstub.py` 中增加一个不包含动态 Reshape 的模型：

```python
def make_shape_tensor_chain_model():
    x = value_info("x", ["batch", 2, 3])
    target = value_info("target", [2], TensorProto.INT64)

    index = initializer(
        "index",
        np.asarray(0, dtype=np.int64),
    )
    axes = initializer(
        "axes",
        np.asarray([0], dtype=np.int64),
    )
    six32 = initializer(
        "six32",
        np.asarray([6], dtype=np.int32),
    )

    nodes = [
        helper.make_node(
            "Shape",
            ["x"],
            ["x_shape"],
        ),
        helper.make_node(
            "Gather",
            ["x_shape", "index"],
            ["batch"],
            axis=0,
        ),
        helper.make_node(
            "Unsqueeze",
            ["batch", "axes"],
            ["batch_1d"],
        ),
        helper.make_node(
            "Cast",
            ["six32"],
            ["six64"],
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Concat",
            ["batch_1d", "six64"],
            ["target"],
            axis=0,
        ),
    ]

    return make_model(
        nodes,
        [x],
        [target],
        [index, axes, six32],
    )
```

增加测试：

```python
class TestShapeTensorOperators(unittest.TestCase):
    def test_shape_tensor_chain_on_native_cpu(self):
        stub = import_model(make_shape_tensor_chain_model())

        for batch in [1, 2, 8, 3, 1]:
            stub.set_input([[batch, 2, 3]])

            # Shape 算子不读取 X.data，但正常调用场景仍应提供输入数据。
            x = np.zeros((batch, 2, 3), dtype=np.float32)
            stub.inputs["x"].copyin_numpy(x)
            stub.run()

            actual = stub.outputs["target"].copyout_numpy()

            self.assertEqual(actual.dtype, np.int64)
            np.testing.assert_array_equal(
                actual,
                np.asarray([batch, 6], dtype=np.int64),
            )
```

这个测试证明：

```text
ONNX importer
  -> ShapeObj/GatherObj/UnsqueezeObj/CastObj/ConcatObj
  -> KernelRegistry
  -> Native CPU Kernel
  -> Int64 Shape Tensor [batch, 6]
```

同一个 `OnnxStub` 连续运行不同 batch，也能确认 `Shape` 每次读取的是 Tensor 当前元数据，而不是首次构图的 `[1,2,3]`。

## 13. 推荐实现顺序

### 阶段 A：只实现 `Shape`

1. 补 `ShapeObj::inferDataType()`；
2. 补 workload/attr；
3. 新增 `ShapeCpu`；
4. 跑 `Shape.InferShapeAndDataType`；
5. 写一个只含 `Shape` 的 Native CPU 数值测试。

完成标准：

```text
X.getDims() = [4,2,3]
Shape output.getDims() = [3]
Shape output.dtype = Int64
Shape output.data = [4,2,3]
```

### 阶段 B：实现 `Gather`

1. 去掉 Operator 对 indices 数据的依赖；
2. 新增 Native CPU Kernel；
3. 测 scalar index；
4. 测一维 indices；
5. 测负 index。

完成标准：

```text
Gather([4,2,3], 0)  = 4
Gather([4,2,3], -1) = 3
```

### 阶段 C：接通 `Unsqueeze/Squeeze/Concat/Cast`

1. 修复 axes 成员副作用；
2. 给 Concat 增加 Int64；
3. 新增 CastCpu；
4. 跑完整 C++ Shape Tensor 链。

完成标准：最终得到 Int64 `[4,6]`。

### 阶段 D：ONNX 端到端

1. 明确拒绝 `Shape start/end`；
2. 构造 Shape Tensor ONNX 模型；
3. 连续运行 `1 → 2 → 8 → 3 → 1`；
4. 每次验证输出为 `[batch,6]`；
5. 跑全部 ONNX importer 回归。

## 14. 构建和运行测试

项目主要面向 Linux/WSL。进入仓库：

```bash
cd /mnt/d/InfiniTensor/Advance/InfiniTensor
```

重新 configure 并构建 CPU Debug 版本：

```bash
cmake -S . -B build/Debug \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TEST=ON \
    -DBUILD_TEST_CORE=ON \
    -DUSE_CUDA=OFF

cmake --build build/Debug -j8
```

只运行新增测试：

```bash
./build/Debug/test_shape
./build/Debug/test_nativecpu_shape_tensor
```

使用 CTest：

```bash
ctest --test-dir build/Debug \
    -R "test_shape|test_nativecpu_shape_tensor" \
    --output-on-failure
```

安装刚编译的 Python backend：

```bash
make install-python TYPE=Debug CUDA=OFF TEST=ON
```

只运行 Shape Tensor Python 测试：

```bash
python3 -m pytest \
    pyinfinitensor/tests/test_onnxstub.py \
    -k "ShapeTensorOperators" \
    -q
```

最后执行回归：

```bash
ctest --test-dir build/Debug --output-on-failure
python3 pyinfinitensor/tests/test_onnxstub.py
python3 pyinfinitensor/tests/test_onnx.py
```

格式化：

```bash
python3 scripts/format.py
```

## 15. 常见错误和定位方式

### 15.1 `Shape` 输出 dtype 变成 Float32

原因：没有覆盖 `ShapeObj::inferDataType()`，默认继承第一个输入 dtype。

### 15.2 `Shape` Kernel 注册了但执行前 halt

原因：没有实现 `ShapeObj::getWorkloadVector()` 或 `getOpAttrVector()`。

### 15.3 `Kernel not found` 或注册表查找失败

确认：

```cpp
REGISTER_KERNEL(Device::CPU, OpType::Shape, ...)
REGISTER_KERNEL(Device::CPU, OpType::Gather, ...)
REGISTER_KERNEL(Device::CPU, OpType::Cast, ...)
```

并在新增 `.cc` 后重新运行 CMake configure。

### 15.4 `Concat` 在 Float32 测试通过，Shape 链失败

检查 `src/kernels/cpu/concat.cc` 是否增加了 `CASE(7)`。

### 15.5 Gather 构图通过但运行越界

indices 是数据依赖，Kernel 必须检查：

```text
-axisDim <= index < axisDim
```

负 index 需要归一化：

```cpp
if (index < 0)
    index += axisDim;
```

### 15.6 Python 测试仍使用旧 backend

修改 C++ 后必须重新构建并安装：

```bash
make install-python TYPE=Debug CUDA=OFF TEST=ON
```

否则 Python 可能仍在加载上一次的 `.so`。

### 15.7 任务一代码影响导入测试

开始本任务前，先确保动态维度导入的基础测试通过，尤其检查：

```python
return tuple(dimensions)  # 不是 tuple[dimensions]
```

以及 `set_input()` 中：

```python
zip(self.inputs, inputShapes)  # 不是 zep
```

否则 Shape Tensor 测试可能在进入算子分支之前就失败。

## 16. 本阶段完成判定

- [ ] `ShapeObj` 输出元数据 Shape 为 `[input rank]`；
- [ ] `ShapeObj` 输出 dtype 为 `Int64`；
- [ ] Native CPU `Shape` 输出实际维度值；
- [ ] `GatherObj` Shape 推导不读取 indices Blob；
- [ ] Native CPU `Gather` 支持 Int32/Int64 indices；
- [ ] Native CPU `Gather` 支持合法负 index；
- [ ] `Unsqueeze/Squeeze` 能处理 Int64 Shape Tensor；
- [ ] `Unsqueeze/Squeeze::inferShape()` 重复调用无成员副作用；
- [ ] Native CPU `Concat` 支持 Int64；
- [ ] Native CPU `Cast` 支持模型实际需要的整数转换；
- [ ] C++ 完整链路得到 Int64 `[batch,6]`；
- [ ] ONNX 完整链路得到 Int64 `[batch,6]`；
- [ ] 同一个模型实例连续执行 `1 → 2 → 8 → 3 → 1`；
- [ ] 现有 Operator、Native CPU 和 ONNX importer 测试无回归。

完成这些之后，再进入下一阶段：让该 Shape Tensor `[batch,6]` 真正成为 `Reshape` 的第二个运行时输入。
