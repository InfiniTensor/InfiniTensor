# Shape Tensor 算子与动态 Reshape 代码修改指南

## 1. 这份方案要完成什么

本方案对应两个任务：

1. 支持项目所需的基础 Shape Tensor 相关算子；
2. 支持由运行时 Shape Tensor 驱动的 `Reshape`。

目标数据流如下：

```text
X: [batch, 2, 3]
 │
 ├─ Shape(X)                    -> [batch, 2, 3]，dtype=Int64
 │    └─ Gather(index=0)        -> batch
 │          └─ Unsqueeze([0])   -> [batch]
 │
 ├─ Constant([6])
 │
 └─ Concat([batch], [6])       -> [batch, 6]
          └─ Reshape(X, shape)  -> Y: [batch, 6]
```

同一个模型实例应能依次执行：

```text
batch = 1 -> 2 -> 8 -> 3 -> 1
```

第一版建议明确限定为：

- 后端：`Native CPU`；
- 动态的是维度值，Tensor rank 在模型生命周期内固定；
- `Shape` 先支持完整输出，不支持 `start/end`；
- `Reshape` 先支持 ONNX 默认的 `allowzero=0`；
- Shape 子图白名单先支持 `Shape/Gather/Unsqueeze/Squeeze/Concat/Cast/Identity`；
- Shape Tensor 的 dtype 按 ONNX 约束为 `Int64`，`Gather` 的 indices 允许 `Int32/Int64`。

这不是只修改 `ReshapeObj` 就能完成。当前 Runtime 在执行前已经为整张图完成内存规划，而动态 `Reshape` 的输出 Shape 要等 Shape 子图运行后才能确定。因此必须增加一次运行前的 Shape 阶段：

```text
已有占位 Shape 完成第一次分配
             ↓
执行 Reshape 所依赖的 Shape 子图
             ↓
读取 Reshape 的 shape Tensor 数据
             ↓
更新 Reshape 及其后继 Tensor 的 Shape 元数据
             ↓
按新 Shape 重新规划内存
             ↓
从头执行完整数据图
```

## 2. 需要修改和新增的文件

| 文件 | 修改目的 |
|---|---|
| `include/operators/unary.h` | 补全 `ShapeObj` 的 dtype、workload 和 attr 接口 |
| `src/operators/unary.cc` | `Shape` 输出固定为 `Int64`；修正 `Cast` 的性能键 |
| `src/kernels/cpu/shape.cc` | 新增 Native CPU `Shape` Kernel |
| `src/operators/gather.cc` | 允许 ONNX 合法的负 indices |
| `src/kernels/cpu/gather.cc` | 新增 Native CPU `Gather` Kernel，至少支持 Int64 Shape Tensor |
| `src/kernels/cpu/concat.cc` | 为 Shape Tensor 增加 `Int32/Int64` 分支 |
| `src/kernels/cpu/cast.cc` | 新增 Native CPU `Cast` Kernel |
| `include/operators/reshape.h` | 保留静态构造函数，新增 Tensor 驱动构造函数和运行时解析接口 |
| `src/operators/reshape.cc` | 统一实现 ONNX Reshape 规则和运行时 Shape Tensor 解析 |
| `include/core/graph.h` | 增加按当前 allocator 模式重新分配的入口 |
| `src/core/graph.cc` | 实现重新推导和重新分配辅助函数 |
| `src/core/runtime.cc` | 增加 Shape 子图预执行、动态 Shape 解析、二次内存规划 |
| `include/core/graph_handler.h` | 增加 `reshapeDynamic` 前端入口 |
| `src/core/graph_handler.cc` | 把两个 Tensor 输入接入 `ReshapeObj` |
| `src/ffi/ffi_infinitensor.cc` | 向 Python 暴露 `reshape_dynamic`，并区分静态/动态 Reshape |
| `pyinfinitensor/src/pyinfinitensor/onnx.py` | ONNX `Reshape` 第二输入非 initializer 时走动态路径 |
| `test/operators/test_reshape.cc` | 补动态 Reshape 的构图与 Shape 解析测试 |
| `test/kernels/nativecpu/test_nativecpu_shape_tensor.cc` | 测基础 Shape Tensor Kernel 与完整动态链路 |
| `pyinfinitensor/tests/test_onnxstub.py` | 测 ONNX 导入和连续五组动态输入 |

项目使用 `file(GLOB_RECURSE ...)` 收集 `src/kernels/cpu/*.cc`，使用 `build_test(test/kernels/nativecpu/*.cc)` 收集测试，因此按上述路径新增 `.cc` 文件通常不需要修改 `CMakeLists.txt`。新增文件后需要重新运行一次 CMake configure，保证 glob 刷新。

## 3. 第一步：补全基础 Shape Tensor 算子

### 3.1 补全 `ShapeObj`

当前问题：

- `ShapeObj::inferShape()` 已能得到输出 Tensor 的元数据 Shape `[input_rank]`；
- 但它没有覆盖 `inferDataType()`，所以会错误继承输入 dtype；
- 没有 CPU Kernel，无法把输入 Tensor 的实际 Shape 写入输出数据；
- 没有实现 workload/attr，Runtime 调用 `getOpPerfKey()` 时会进入默认的 `IT_TODO_HALT()`。

#### 修改 `include/operators/unary.h`

把 `ShapeObj` 改为：

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

#### 修改 `src/operators/unary.cc`

在现有 `ShapeObj` 实现后补上：

```cpp
vector<DataType> ShapeObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 1);
    return {DataType::Int64};
}

vector<int> ShapeObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const auto &inputShape = inputs[0]->getDims();
    ret.insert(ret.end(), inputShape.begin(), inputShape.end());
    return ret;
}

vector<int> ShapeObj::getOpAttrVector() const {
    return {type.underlying()};
}
```

现有 `inferShape()` 保留：

```cpp
optional<vector<Shape>> ShapeObj::inferShape(const TensorVec &inputs) {
    return {{{static_cast<int>(inputs[0]->getRank())}}};
}
```

注意这里的两种 Shape：

```text
输入 X.getDims() = [4, 2, 3]
Shape 输出 S.getDims() = [3]       // S 自己是一维 Tensor
Shape 输出 S.data = [4, 2, 3]      // Kernel 写入的数据
```

#### 新增 `src/kernels/cpu/shape.cc`

完整代码：

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

        const auto &dims = op->getInputs(0)->getDims();
        auto *output = op->getOutput()->getRawDataPtr<int64_t *>();
        for (size_t i = 0; i < dims.size(); ++i)
            output[i] = static_cast<int64_t>(dims[i]);
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Shape, ShapeCpu, "Shape_CPU");

} // namespace infini
```

### 3.2 新增 Native CPU `Gather`

当前 `GatherObj` 的 Shape/dtype 推导基本可复用，但 `CheckIndexValid()` 错误拒绝负 index。ONNX `Gather` 的合法范围是：

```text
[-axisDim, axisDim - 1]
```

#### 修改 `src/operators/gather.cc`

将两处检查：

```cpp
if (data[i] < 0 || data[i] >= value)
```

都改为：

```cpp
if (data[i] < -value || data[i] >= value)
```

更重要的是，Kernel 必须在执行时再次检查，因为动态 indices 在构图时没有数据。

#### 新增 `src/kernels/cpu/gather.cc`

完整代码：

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

        const auto data = op->getInputs(0);
        const auto indices = op->getInputs(1);
        const auto output = op->getOutput();
        const auto &dims = data->getDims();
        const int axis = op->getAxis();

        size_t outer = 1;
        size_t inner = 1;
        for (int i = 0; i < axis; ++i)
            outer *= static_cast<size_t>(dims[i]);
        for (size_t i = static_cast<size_t>(axis + 1); i < dims.size(); ++i)
            inner *= static_cast<size_t>(dims[i]);

        const int64_t axisDim = dims[axis];
        const size_t indexCount = indices->size();
        const size_t elementBytes = data->getDType().getSize();
        const size_t blockBytes = inner * elementBytes;

        const auto *indexData = indices->getRawDataPtr<const IndexT *>();
        const auto *input = data->getRawDataPtr<const uint8_t *>();
        auto *out = output->getRawDataPtr<uint8_t *>();

        for (size_t o = 0; o < outer; ++o) {
            for (size_t j = 0; j < indexCount; ++j) {
                int64_t index = static_cast<int64_t>(indexData[j]);
                IT_ASSERT(index >= -axisDim && index < axisDim,
                          "Gather index is out of range");
                if (index < 0)
                    index += axisDim;

                const size_t inputBlock =
                    (o * static_cast<size_t>(axisDim) +
                     static_cast<size_t>(index)) *
                    inner;
                const size_t outputBlock = (o * indexCount + j) * inner;
                std::memcpy(out + outputBlock * elementBytes,
                            input + inputBlock * elementBytes, blockBytes);
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        const auto dtype = _op->getInputs(1)->getDType();
        if (dtype == DataType::Int32)
            doCompute<int32_t>(_op);
        else if (dtype == DataType::Int64)
            doCompute<int64_t>(_op);
        else
            IT_ASSERT(false, "Gather indices must be Int32 or Int64");
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Gather, GatherCpu, "Gather_CPU");

} // namespace infini
```

这个实现按字节复制 data 元素，因此不用为 `Float32/Int32/Int64` 分别实例化数据类型；只需要分派 indices 类型。

### 3.3 让 `Concat` 支持 Int64 Shape Tensor

当前 `src/kernels/cpu/concat.cc` 只分派 `Float32` 和 `UInt32`，Shape 子图在这里会失败。

把 `NaiveConcat::compute()` 中的 switch 改为：

```cpp
int dataTypeIdx = _op->getDType().getIndex();
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

`ConcatObj::inferShape()` 还应补一个 dtype 一致性检查，避免把不同 dtype 的 Shape Tensor 拼到一起。在 `src/operators/concat.cc` 的 `inferShape()` 开头加入：

```cpp
IT_ASSERT(!inputs.empty());
const auto dtype = inputs[0]->getDType();
for (const auto &input : inputs)
    IT_ASSERT(input->getDType() == dtype,
              "Concat inputs must have the same dtype");
```

### 3.4 `Unsqueeze/Squeeze` 暂时不需要新 Kernel

当前 Native CPU 把 `Unsqueeze/Squeeze/Reshape/Flatten/Identity` 都注册到 `NaiveIdentity`：

```cpp
REGISTER_KERNEL(Device::CPU, OpType::Unsqueeze, NaiveIdentity, ...);
REGISTER_KERNEL(Device::CPU, OpType::Squeeze, NaiveIdentity, ...);
```

这些算子只改变 Shape 元数据，不改变元素顺序，因此字节复制对 Int64 Shape Tensor 也是正确的。第一版无需新增 Kernel。

但应修复两个 Operator 的副作用：当前 `inferShape()` 会修改成员 `axes`。连续调用 `shape_infer()` 时不应反复改属性。建议用局部变量。

`src/operators/unsqueeze.cc`：

```cpp
optional<vector<Shape>> UnsqueezeObj::inferShape(const TensorVec &inputs) {
    const Shape inputDim = inputs[0]->getDims();
    const int rank = static_cast<int>(inputs[0]->getRank() + axes.size());
    Shape normalizedAxes = axes;
    Shape outputShape(rank, -1);

    for (auto &axis : normalizedAxes) {
        axis = get_real_axis(axis, rank);
        IT_ASSERT(outputShape[axis] == -1, "Axes have duplicate");
        outputShape[axis] = 1;
    }

    auto it = inputDim.begin();
    for (auto &dim : outputShape) {
        if (dim == -1)
            dim = *it++;
    }
    return {{outputShape}};
}
```

`src/operators/squeeze.cc` 中也只修改局部 `normalizedAxes`，不要在 `axes.empty()` 时向成员 `axes` 追加内容：

```cpp
optional<vector<Shape>> SqueezeObj::inferShape(const TensorVec &inputs) {
    const Shape inputDim = inputs[0]->getDims();
    const int rank = inputs[0]->getRank();
    Shape normalizedAxes = axes;

    if (normalizedAxes.empty()) {
        for (int i = 0; i < rank; ++i)
            if (inputDim[i] == 1)
                normalizedAxes.emplace_back(i);
    }

    for (auto &axis : normalizedAxes) {
        axis = get_real_axis(axis, rank);
        IT_ASSERT(inputDim[axis] == 1,
                  "Squeeze axis must refer to a dimension of size 1");
    }
    std::sort(normalizedAxes.begin(), normalizedAxes.end());
    IT_ASSERT(std::adjacent_find(normalizedAxes.begin(), normalizedAxes.end()) ==
                  normalizedAxes.end(),
              "Squeeze axes have duplicate");

    Shape outputShape;
    for (int i = 0; i < rank; ++i) {
        if (!std::binary_search(normalizedAxes.begin(), normalizedAxes.end(), i))
            outputShape.emplace_back(inputDim[i]);
    }
    return {{outputShape}};
}
```

### 3.5 新增 Native CPU `Cast`

Shape 子图最常用的是 `Int32 <-> Int64`。建议第一版至少支持下面四种：

```text
Float32 -> Int32
Float32 -> Int64
Int32   -> Int64
Int64   -> Int32
```

#### 修正 `src/operators/unary.cc` 中 Cast 的性能键

当前 `getOpAttrVector()` 没有包含 `castType`。改为：

```cpp
vector<int> CastObj::getWorkloadVector() const {
    vector<int> ret{type.underlying(), static_cast<int>(castType)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> CastObj::getOpAttrVector() const {
    return {type.underlying(), static_cast<int>(castType)};
}
```

#### 新增 `src/kernels/cpu/cast.cc`

完整代码：

```cpp
#include "core/kernel.h"
#include "operators/unary.h"

namespace infini {

class CastCpu final : public CpuKernelWithoutConfig {
    template <typename InputT, typename OutputT>
    void cast(const Operator &_op) const {
        auto op = as<CastObj>(_op);
        const auto *input = op->getInputs(0)->getRawDataPtr<const InputT *>();
        auto *output = op->getOutput()->getRawDataPtr<OutputT *>();
        for (size_t i = 0; i < op->getOutput()->size(); ++i)
            output[i] = static_cast<OutputT>(input[i]);
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<CastObj>(_op);
        IT_ASSERT(op != nullptr);
        switch (op->getType()) {
        case CastType::Float2Int32:
            cast<float, int32_t>(_op);
            break;
        case CastType::Float2Int64:
            cast<float, int64_t>(_op);
            break;
        case CastType::Int322Int64:
            cast<int32_t, int64_t>(_op);
            break;
        case CastType::Int642Int32:
            cast<int64_t, int32_t>(_op);
            break;
        case CastType::Int642Float:
            cast<int64_t, float>(_op);
            break;
        case CastType::Int322Float:
            cast<int32_t, float>(_op);
            break;
        case CastType::Float2Float:
            cast<float, float>(_op);
            break;
        default:
            IT_ASSERT(false, "Cast type is not supported by Native CPU");
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Cast, CastCpu, "Cast_CPU");

} // namespace infini
```

## 4. 第二步：让 `Reshape` 同时支持静态 Shape 和 Shape Tensor

### 4.1 接口设计

不要删除现有静态接口：

```cpp
ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims);
```

新增动态接口：

```cpp
ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor,
           Tensor output, bool allowZero = false);
```

内部统一维护：

```cpp
bool runtimeShape; // false=静态属性，true=第二个输入 Tensor
bool allowZero;
Shape dims;        // 静态 Reshape 的原始规格
Shape outputShape; // 当前已经解析出的实际输出 Shape
```

### 4.2 修改 `include/operators/reshape.h`

将 `ReshapeObj` 替换为：

```cpp
class ReshapeObj : public OperatorObj {
    Shape dims;
    Shape outputShape;
    bool runtimeShape = false;
    bool allowZero = false;

  public:
    // 保留现有静态接口。
    ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims);

    // 新增由第二个 Tensor 输入驱动的接口。
    ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor,
               Tensor output, bool allowZero = false);

    OP_CLONE(ReshapeObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    int numInputs() const override { return static_cast<int>(inputs.size()); }
    int numOutputs() const override { return 1; }

    bool isRuntimeShape() const { return runtimeShape; }
    bool getAllowZero() const { return allowZero; }
    Tensor getShapeTensor() const {
        return runtimeShape ? inputs.at(1) : nullptr;
    }

    // Shape 阶段调用：读取 inputs[1].data，更新 outputShape 和输出元数据。
    bool resolveRuntimeShape();

    Shape getShape() const { return outputShape; }
    Shape getDims() const { return dims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
```

### 4.3 修改 `src/operators/reshape.cc`

建议把文件中 `ReshapeObj` 相关实现替换为下面代码；后面的 `FlattenObj` 和 `IdentityObj` 保留。

```cpp
#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <limits>
#include <numeric>

namespace infini {
namespace {

Shape resolveReshapeSpec(const Shape &inputShape, size_t inputSize,
                         const vector<int64_t> &spec, bool allowZero) {
    Shape result(spec.size());
    int inferAxis = -1;
    size_t knownProduct = 1;

    for (size_t i = 0; i < spec.size(); ++i) {
        int64_t value = spec[i];
        IT_ASSERT(value >= -1, "Reshape dimensions must be >= -1");

        if (value == -1) {
            IT_ASSERT(inferAxis == -1,
                      "Reshape can contain at most one -1");
            inferAxis = static_cast<int>(i);
            result[i] = -1;
            continue;
        }

        if (value == 0 && !allowZero) {
            IT_ASSERT(i < inputShape.size(),
                      "Reshape 0 dimension exceeds input rank");
            value = inputShape[i];
        }

        IT_ASSERT(value <= std::numeric_limits<int>::max(),
                  "Reshape dimension exceeds InfiniTensor Shape range");
        result[i] = static_cast<int>(value);

        if (result[i] == 0) {
            knownProduct = 0;
        } else if (knownProduct != 0) {
            IT_ASSERT(knownProduct <=
                          std::numeric_limits<size_t>::max() /
                              static_cast<size_t>(result[i]),
                      "Reshape element count overflow");
            knownProduct *= static_cast<size_t>(result[i]);
        }
    }

    if (inferAxis >= 0) {
        IT_ASSERT(knownProduct != 0,
                  "Reshape with -1 and a zero-sized known product is ambiguous");
        IT_ASSERT(inputSize % knownProduct == 0,
                  "Reshape -1 dimension cannot be inferred exactly");
        const size_t inferred = inputSize / knownProduct;
        IT_ASSERT(inferred <= static_cast<size_t>(std::numeric_limits<int>::max()),
                  "Inferred Reshape dimension is too large");
        result[inferAxis] = static_cast<int>(inferred);
    } else {
        IT_ASSERT(knownProduct == inputSize,
                  "Reshape input and output element counts differ");
    }

    if (allowZero && inferAxis >= 0) {
        // ONNX 明确禁止 allowzero=1 时同时使用 0 和 -1。
        IT_ASSERT(std::find(result.begin(), result.end(), 0) == result.end(),
                  "Reshape allowzero=1 cannot combine 0 and -1");
    }
    return result;
}

vector<int64_t> toInt64(const Shape &shape) {
    return vector<int64_t>(shape.begin(), shape.end());
}

} // namespace

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}), dims(std::move(dims)),
      runtimeShape(false), allowZero(false) {
    IT_ASSERT(checkValid(graph));
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor,
                       Tensor output, bool allowZero)
    : OperatorObj(OpType::Reshape, {input, shapeTensor}, {output}),
      outputShape(output ? output->getDims()
                         : Shape(shapeTensor->size(), 1)),
      runtimeShape(true), allowZero(allowZero) {
    IT_ASSERT(shapeTensor->getRank() == 1,
              "Reshape shape input must be a 1-D Tensor");
    IT_ASSERT(shapeTensor->getDType() == DataType::Int64,
              "ONNX Reshape shape input must be Int64");
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    if (runtimeShape) {
        // inferShape 只传播最近一次已经解析出的结果，绝不在普通 Shape
        // inference 中读取可能尚未执行或属于上一次 run 的 Tensor 数据。
        IT_ASSERT(inputs.size() == 2);
        return {{outputShape}};
    }

    outputShape = resolveReshapeSpec(inputs[0]->getDims(), inputs[0]->size(),
                                     toInt64(dims), allowZero);
    return {{outputShape}};
}

bool ReshapeObj::resolveRuntimeShape() {
    IT_ASSERT(runtimeShape);
    const auto shapeTensor = inputs[1];
    IT_ASSERT(shapeTensor->hasData(),
              "Runtime Reshape shape Tensor has no data");
    IT_ASSERT(shapeTensor->getDataBlob()->getBytes() ==
                  shapeTensor->getBytes(),
              "Runtime Reshape shape Tensor storage is invalid");

    const vector<int64_t> spec = shapeTensor->copyout<int64_t>();
    Shape resolved = resolveReshapeSpec(inputs[0]->getDims(), inputs[0]->size(),
                                        spec, allowZero);
    const bool changed = resolved != outputShape;
    outputShape = std::move(resolved);
    outputs[0]->setShape(outputShape);
    return changed;
}

std::string ReshapeObj::toString() const {
    std::ostringstream os;
    os << "Reshape[" << getGuid() << "](";
    os << "inputShape=" << vecToString(inputs[0]->getDims()) << ",";
    os << "outputShape=" << vecToString(outputShape) << ",";
    os << "runtimeShape=" << runtimeShape << ",";
    if (runtimeShape)
        os << "shapeTensor=" << inputs[1]->getGuid() << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReshapeObj::getWorkloadVector() const {
    vector<int> ret{type.underlying(), static_cast<int>(runtimeShape),
                    static_cast<int>(allowZero)};
    const Shape inputShape = inputs[0]->getDims();
    ret.insert(ret.end(), inputShape.begin(), inputShape.end());
    ret.insert(ret.end(), outputShape.begin(), outputShape.end());
    return ret;
}

vector<int> ReshapeObj::getOpAttrVector() const {
    vector<int> ret{type.underlying(), static_cast<int>(runtimeShape),
                    static_cast<int>(allowZero)};
    if (!runtimeShape)
        ret.insert(ret.end(), dims.begin(), dims.end());
    return ret;
}

// FlattenObj 和 IdentityObj 的原有实现继续放在这里。
```

上面代码需要 `<algorithm>`，项目公共头通常已经间接包含；为避免依赖间接 include，建议显式加入：

```cpp
#include <algorithm>
```

### 4.4 为什么动态 `inferShape()` 不能直接读 Shape Tensor

下面这种实现是错误的：

```cpp
optional<vector<Shape>> ReshapeObj::inferShape(...) {
    return {{inputs[1]->copyout<int64_t>()}}; // 不要这样做
}
```

原因：

- 构图时 Shape 子图尚未执行，Blob 可能不存在；
- `set_input()` 调用普通 `shape_infer()` 时，Blob 可能仍是上一次运行的值；
- Shape 改变后，内存重规划可能更换 Blob；
- Shape Tensor 不一定在 CPU 设备上，直接取指针不具有通用性。

所以必须把“纯元数据推导”和“读取运行时 Shape Tensor 数据”拆成两个接口：

```text
inferShape()          -> 静态传播/传播上次已解析结果
resolveRuntimeShape() -> 仅在 Shape 子图刚执行完成时调用
```

## 5. 第三步：增加运行前 Shape 阶段

### 5.1 为 Graph 增加按原 allocator 模式重分配的接口

`OnnxStub` 可能使用默认动态内存池，也可能使用 naive allocator。Runtime 不能简单调用 `graph->dataMalloc()`，否则原来使用 naive allocator 的图会被判定为切换 allocator 模式。

#### 修改 `include/core/graph.h`

在 `dataMalloc()` 后增加：

```cpp
// Shape 元数据改变后，沿用首次分配时锁定的 allocator 模式重新规划。
void remallocForCurrentShape();
```

#### 修改 `src/core/graph.cc`

在 `dataMalloc()` 后增加：

```cpp
void GraphObj::remallocForCurrentShape() {
    switch (allocationMode) {
    case AllocationMode::Uninitialized:
        dataMalloc(false, 0);
        break;
    case AllocationMode::Naive:
        dataMalloc(true, 0);
        break;
    case AllocationMode::DynamicPool:
        dataMalloc(false, 0);
        break;
    case AllocationMode::FixedPool:
        dataMalloc(false, fixedPoolSize);
        break;
    }
}
```

现有 fixed pool 逻辑会在布局变化时给出：

```text
Fixed memory pool does not support dynamic memory layout changes
```

这是合理的第一版限制，不要绕开检查。训练营动态模型使用默认动态内存池或 naive allocator。

另外建议在 `GraphObj::shape_infer()` 开头保证拓扑顺序：

```cpp
void GraphObj::shape_infer() {
    IT_ASSERT(topo_sort(), "Graph contains a cycle");
    // 原有循环保持不变
}
```

### 5.2 Shape 子图的识别规则

从每个动态 `Reshape` 的第二输入反向遍历 producer：

```text
dynamic Reshape.inputs[1]
        ↑
      Concat
      ↑    ↑
Unsqueeze Constant
    ↑
  Gather
    ↑
  Shape   <- 遍历到 Shape 后停止，不继续追它的数据输入
```

第一版只允许下面的 producer：

```cpp
Shape, Gather, Unsqueeze, Squeeze, Concat, Cast, Identity
```

到达 `Shape` 时必须停止反向追踪，因为 `Shape` 只读输入 Tensor 的 Shape 元数据，不需要计算其数据。否则可能把前面整段神经网络计算误判成 Shape 子图。

如果 Shape Tensor 依赖 `Add/Mul/NonZero` 等不在白名单的算子，应明确报错，而不是静默执行错误结果。后续根据真实模型再扩白名单。

### 5.3 修改 `src/core/runtime.cc`

先加入：

```cpp
#include "operators/reshape.h"
#include <unordered_set>
```

在 `namespace infini` 内、`CpuRuntimeObj::run()` 前增加下面的辅助代码：

```cpp
namespace {

bool isShapeValueOp(OpType type) {
    switch (type.underlying()) {
    case OpType::Shape:
    case OpType::Gather:
    case OpType::Unsqueeze:
    case OpType::Squeeze:
    case OpType::Concat:
    case OpType::Cast:
    case OpType::Identity:
        return true;
    default:
        return false;
    }
}

void collectShapeValueOps(const Tensor &tensor,
                          std::unordered_set<OperatorObj *> &required) {
    const auto source = tensor->getSource();
    if (!source)
        return; // initializer 或由调用者提供的 Shape Tensor 输入

    IT_ASSERT(isShapeValueOp(source->getOpType()),
              "Runtime Reshape shape input depends on an unsupported "
              "Shape Tensor operator: " +
                  std::string(source->getOpType().toString()));

    if (!required.insert(source.get()).second)
        return;

    if (source->getOpType() == OpType::Shape)
        return; // Shape 只依赖输入元数据，不需要输入数据

    for (const auto &input : source->getInputs())
        collectShapeValueOps(input, required);
}

bool prepareRuntimeShapes(const Graph &graph, const CpuRuntimeObj *runtime) {
    vector<Ref<ReshapeObj>> dynamicReshapes;
    std::unordered_set<OperatorObj *> requiredShapeOps;

    IT_ASSERT(graph->topo_sort(), "Graph contains a cycle");
    for (const auto &op : graph->getOperators()) {
        if (op->getOpType() != OpType::Reshape)
            continue;
        auto reshape = as<ReshapeObj>(op);
        if (!reshape->isRuntimeShape())
            continue;
        dynamicReshapes.emplace_back(reshape);
        collectShapeValueOps(reshape->getShapeTensor(), requiredShapeOps);
    }

    if (dynamicReshapes.empty())
        return false;

    // 第一版只承诺 Native CPU。MklRuntime 也继承 CpuRuntimeObj，但其
    // Device 不是 CPU；等 IntelCPU Kernel 补齐后再放开。
    IT_ASSERT(runtime->getDevice() == Device::CPU,
              "Runtime Shape Tensor execution currently supports Native CPU only");

    const auto &registry = KernelRegistry::getInstance();

    // graph 已经是拓扑序。先算 Shape 值；遇到动态 Reshape 时只更新
    // Shape 元数据，不执行数据复制。
    for (const auto &op : graph->getOperators()) {
        if (requiredShapeOps.count(op.get())) {
            const KernelAttrs attrs{Device::CPU,
                                    op->getOpType().underlying()};
            Kernel *kernel = registry.getKernel(attrs);
            kernel->compute(op, runtime);
        }

        if (op->getOpType() == OpType::Reshape) {
            auto reshape = as<ReshapeObj>(op);
            if (reshape->isRuntimeShape()) {
                reshape->resolveRuntimeShape();
                // 让普通后继算子的元数据立即看到新的 Reshape 输出 Shape。
                graph->shape_infer();
            }
        }
    }

    graph->shape_infer();
    graph->remallocForCurrentShape();
    graph->validateMemory();
    return true;
}

} // namespace
```

`RuntimeObj` 当前可能没有公开 `getDevice()`。若没有，在 `include/core/runtime.h` 的 `RuntimeObj` public 区加入：

```cpp
Device getDevice() const { return device; }
```

然后在 `CpuRuntimeObj::run()` 的开头，把：

```cpp
graph->validateMemory();
```

改为：

```cpp
graph->validateMemory();
prepareRuntimeShapes(graph, this);
```

后面的完整算子循环不删。二次内存规划会使 Shape 中间值失效，但完整循环会从 `Shape` 开始重新计算它们，然后动态 `Reshape` 的 CPU Kernel 仍然只复制第一个 data 输入。

最终一次 `run()` 的真实过程是：

```text
prepareRuntimeShapes:
  Shape -> Gather -> Unsqueeze -> Concat
  -> resolveRuntimeShape
  -> shape_infer
  -> remallocForCurrentShape

原有完整执行循环:
  Shape -> Gather -> Unsqueeze -> Concat
  -> Reshape memcpy
  -> 后续数据算子
```

### 5.4 为什么不能在执行到 Reshape 时才临时重分配

不要采用下面的单遍方案：

```text
执行一部分数据算子
  -> 遇到动态 Reshape
  -> 更新 Shape
  -> 整图 dataMalloc
  -> 继续执行
```

整图重规划可能移动或覆盖已经计算好的中间 Tensor。现有 allocator 只专门保留图输入和权重，不保证保存所有中间激活值。因此必须在任何数据计算之前完成 Shape 解析和最终内存规划。

## 6. 第四步：接通 GraphHandler、FFI 和 ONNX 导入

### 6.1 修改 `include/core/graph_handler.h`

在静态 `reshape` 后增加：

```cpp
Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
Tensor reshapeDynamic(Tensor data, Tensor shapeTensor, Tensor reshaped,
                      bool allowZero = false);
```

### 6.2 修改 `src/core/graph_handler.cc`

在现有 `GraphHandlerObj::reshape()` 后增加：

```cpp
Tensor GraphHandlerObj::reshapeDynamic(Tensor data, Tensor shapeTensor,
                                       Tensor reshaped, bool allowZero) {
    if (reshaped) {
        g->addOpWithOutputs<ReshapeObj>(std::move(data),
                                        std::move(shapeTensor), reshaped,
                                        allowZero);
        return reshaped;
    }
    return g
        ->addOp<ReshapeObj>(std::move(data), std::move(shapeTensor), reshaped,
                            allowZero)
        ->getOutput();
}
```

### 6.3 修改 `src/ffi/ffi_infinitensor.cc`

在 GraphHandler 的 pybind 注册处增加：

```cpp
.def("reshape_dynamic", &Handler::reshapeDynamic, policy::move)
```

为了让 `to_onnx()` 不给动态 Reshape 再伪造一个静态 initializer，增加查询函数：

```cpp
static bool reshape_is_dynamic_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Reshape);
    return dynamic_cast<const ReshapeObj *>(op.get())->isRuntimeShape();
}
```

把它加入已有的 `.FUNCTION(...)` 注册链：

```cpp
.FUNCTION(reshape_is_dynamic_of)
```

### 6.4 修改 ONNX importer

文件：`pyinfinitensor/src/pyinfinitensor/onnx.py`

将当前 `Reshape` 分支：

```python
elif node.op_type == "Reshape":
    shape = _parse_static_input(data, node, 1, required=True)
    tensors[node.output[0]] = self.handler.reshape(
        tensors[node.input[0]],
        tensors.get(node.output[0]),
        shape,
    )
```

替换为：

```python
elif node.op_type == "Reshape":
    if not _has_input(node, 1):
        raise ValueError("Reshape requires a shape input")

    allowzero = next(
        (attr.i for attr in node.attribute if attr.name == "allowzero"),
        0,
    )
    if allowzero != 0:
        raise NotImplementedError(
            "Reshape allowzero=1 is not supported in the first implementation"
        )

    shape_name = node.input[1]
    if shape_name in data:
        # 保持旧的静态路径，避免静态模型行为和导出格式发生变化。
        tensors[node.output[0]] = self.handler.reshape(
            tensors[node.input[0]],
            tensors.get(node.output[0]),
            _parse_data(data[shape_name]),
        )
    else:
        # 第二输入来自 Shape/Gather/Concat 等运行时子图。
        tensors[node.output[0]] = self.handler.reshape_dynamic(
            tensors[node.input[0]],
            tensors[shape_name],
            tensors.get(node.output[0]),
            False,
        )
```

`Shape` 第一版不支持 `start/end`，不要静默忽略。将分支改为：

```python
elif node.op_type == "Shape":
    attributes = _parse_attribute(node, {"start": 0, "end": None})
    if attributes["start"] != 0 or attributes["end"] is not None:
        raise NotImplementedError(
            "Shape start/end slicing is not supported yet"
        )
    tensors[node.output[0]] = self.handler.shape(
        tensors[node.input[0]],
        tensors.get(node.output[0]),
    )
```

### 6.5 修改 `to_onnx()` 的 Reshape 导出

当前导出逻辑会无条件追加静态 shape initializer。改为：

```python
elif ty == backend.OpTypeId.Reshape:
    if not backend.reshape_is_dynamic_of(op):
        shape = backend.reshape_shape_of(op)
        inputs.append(
            ctx.push_data_input(
                name,
                "shape",
                TensorProto.INT64,
                [len(shape)],
                shape,
            )
        )
    # 动态 Reshape 的 inputs 已经包含 data 和 shapeTensor，不能再 append。
    ctx.push_node(make_node(ty.name, inputs, outputs, name))
```

## 7. 动态 Reshape 的占位 Shape 问题

构图时运行时 Shape Tensor 尚无数据，但 `OperatorObj::checkValid()` 必须创建输出 Tensor。上面的 C++ 构造函数用：

```cpp
Shape(shapeTensor->size(), 1)
```

作为保底占位 Shape。它只能保证 rank 正确，不一定能让所有严格检查维度关系的后继算子成功构造。

推荐在 importer 中利用 ONNX 的 `graph.value_info` 或 `graph.output` 给动态 Reshape 输出提供更准确的占位元数据。

在 `OnnxStub.__init__()` 中先建立声明信息表：

```python
declared_value_info = {
    value.name: value
    for value in list(model.graph.value_info) + list(model.graph.output)
}
```

动态 Reshape 分支调用前：

```python
output = tensors.get(node.output[0])
declared = declared_value_info.get(node.output[0])
if output is None and declared is not None:
    tensor_type = declared.type.tensor_type
    output = self.handler.tensor(
        _take_shape_dim(tensor_type.shape),
        tensor_type.elem_type,
    )
    tensors[node.output[0]] = output

tensors[node.output[0]] = self.handler.reshape_dynamic(
    tensors[node.input[0]],
    tensors[shape_name],
    output,
    False,
)
```

对于：

```text
Y: ["batch", 6]
```

当前 `_take_shape_dim()` 会物化为 `[1, 6]`，比纯 rank 占位 `[1, 1]` 更适合构造后继算子。运行时仍会用 `[actual_batch, 6]` 覆盖它。

第一版应在文档中声明：动态的是各维数值，target rank 必须固定。若 Shape Tensor 的元素个数会在不同运行间改变，第一次分配本身就可能不足，暂不支持。

## 8. 测试代码

### 8.1 `ShapeObj` Operator 测试

可以放到 `test/operators/test_unary.cc` 或新增 `test/operators/test_shape.cc`：

```cpp
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {

TEST(Shape, InferShapeAndDType) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto x = g->addTensor({4, 2, 3}, DataType::Float32);
    auto op = g->addOp<ShapeObj>(x, nullptr);

    EXPECT_EQ(op->getOutput()->getDims(), (Shape{3}));
    EXPECT_EQ(op->getOutput()->getDType(), DataType::Int64);
}

} // namespace infini
```

### 8.2 Native CPU 基础 Shape Tensor Kernel 测试

新增 `test/kernels/nativecpu/test_nativecpu_shape_tensor.cc`：

```cpp
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/gather.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include "test.h"

namespace infini {

TEST(NativeCpuShapeTensor, ShapeGatherUnsqueezeConcat) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto x = g->addTensor({4, 2, 3}, DataType::Float32);
    x->setInput();
    auto shape = g->addOp<ShapeObj>(x, nullptr)->getOutput();

    auto index = g->addTensor({}, DataType::Int64);
    index->setWeight();
    auto batch = g->addOp<GatherObj>(shape, index, nullptr, 0)->getOutput();
    auto batch1d =
        g->addOp<UnsqueezeObj>(batch, nullptr, Shape{0})->getOutput();

    auto six = g->addTensor({1}, DataType::Int64);
    six->setWeight();
    auto target =
        g->addOp<ConcatObj>(TensorVec{batch1d, six}, nullptr, 0)->getOutput();
    target->setOutput();

    g->dataMalloc();
    index->copyin<int64_t>({0});
    six->copyin<int64_t>({6});
    runtime->run(g);

    EXPECT_TRUE(shape->equalData<int64_t>({4, 2, 3}));
    EXPECT_TRUE(batch->equalData<int64_t>({4}));
    EXPECT_TRUE(batch1d->equalData<int64_t>({4}));
    EXPECT_TRUE(target->equalData<int64_t>({4, 6}));
}

TEST(NativeCpuShapeTensor, GatherAcceptsNegativeIndex) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto data = g->addTensor({3}, DataType::Int64);
    auto index = g->addTensor({}, DataType::Int64);
    data->setInput();
    index->setInput();
    auto output = g->addOp<GatherObj>(data, index, nullptr, 0)->getOutput();
    output->setOutput();

    g->dataMalloc();
    data->copyin<int64_t>({4, 2, 3});
    index->copyin<int64_t>({-1});
    runtime->run(g);
    EXPECT_TRUE(output->equalData<int64_t>({3}));
}

} // namespace infini
```

### 8.3 C++ 完整动态 Reshape 测试

追加到同一个文件：

```cpp
#include "operators/reshape.h"

TEST(NativeCpuDynamicReshape, RuntimeShapeTensorAndRepeatedShapes) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto x = g->addTensor({1, 2, 3}, DataType::Float32);
    x->setInput();
    auto shape = g->addOp<ShapeObj>(x, nullptr)->getOutput();

    auto index = g->addTensor({}, DataType::Int64);
    index->setWeight();
    auto batch = g->addOp<GatherObj>(shape, index, nullptr, 0)->getOutput();
    auto batch1d =
        g->addOp<UnsqueezeObj>(batch, nullptr, Shape{0})->getOutput();

    auto six = g->addTensor({1}, DataType::Int64);
    six->setWeight();
    auto target =
        g->addOp<ConcatObj>(TensorVec{batch1d, six}, nullptr, 0)->getOutput();

    auto reshape =
        g->addOp<ReshapeObj>(x, target, nullptr, false);
    auto y = reshape->getOutput();
    y->setOutput();

    g->dataMalloc();
    index->copyin<int64_t>({0});
    six->copyin<int64_t>({6});

    for (int batchSize : {1, 2, 8, 3, 1}) {
        x->setShape({batchSize, 2, 3});
        g->shape_infer();
        g->remallocForCurrentShape();

        vector<float> input(static_cast<size_t>(batchSize) * 6);
        std::iota(input.begin(), input.end(), 0.0f);
        x->copyin<float>(input);

        runtime->run(g);

        EXPECT_EQ(y->getDims(), (Shape{batchSize, 6}));
        EXPECT_TRUE(y->equalData<float>(input));
    }
}
```

需要在测试文件顶部加入：

```cpp
#include <numeric>
```

### 8.4 ONNX 端到端测试

在 `pyinfinitensor/tests/test_onnxstub.py` 增加：

```python
def make_dynamic_reshape_model():
    x = value_info("x", ["batch", 2, 3])
    y = value_info("y", ["batch", 6])

    index = initializer("index", np.asarray(0, dtype=np.int64))
    axes = initializer("axes", np.asarray([0], dtype=np.int64))
    six = initializer("six", np.asarray([6], dtype=np.int64))

    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Gather", ["x_shape", "index"], ["batch"], axis=0),
        helper.make_node("Unsqueeze", ["batch", "axes"], ["batch_1d"]),
        helper.make_node("Concat", ["batch_1d", "six"], ["target"], axis=0),
        helper.make_node("Reshape", ["x", "target"], ["y"]),
    ]
    return make_model(nodes, [x], [y], [index, axes, six])


class TestDynamicShapeTensor(unittest.TestCase):
    def test_runtime_shape_tensor_drives_reshape_repeatedly(self):
        stub = import_model(make_dynamic_reshape_model())

        for batch in [1, 2, 8, 3, 1]:
            x = np.arange(batch * 6, dtype=np.float32).reshape(batch, 2, 3)
            stub.set_input([[batch, 2, 3]])
            stub.inputs["x"].copyin_numpy(x)
            stub.run()

            actual = stub.outputs["y"].copyout_numpy()
            expected = x.reshape(batch, 6)
            self.assertEqual(list(actual.shape), [batch, 6])
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
```

这个测试一次覆盖：

- ONNX importer 不再要求 `Reshape` 第二输入是 initializer；
- `Shape` 输出 dtype 为 Int64；
- `Shape/Gather/Unsqueeze/Concat` 的 CPU 数值执行；
- Runtime Shape 阶段；
- 动态 `Reshape` 输出 Shape 更新；
- 由小变大和由大变小时的内存重规划；
- 同一个 `OnnxStub` 连续执行，不重新加载模型。

## 9. 推荐实现顺序

不要一次性把 Runtime 和所有算子混在一起改。按下面顺序，每一步都形成可检查结果。

### 阶段 A：先让 Shape Tensor 链在静态图中算对

1. `ShapeObj::inferDataType()`；
2. `Shape` CPU Kernel；
3. `Gather` CPU Kernel和负 index；
4. `Concat` 的 Int64 分派；
5. `Cast` CPU Kernel；
6. 修复 `Squeeze/Unsqueeze::inferShape()` 不修改成员状态；
7. 跑 Operator 和 Native CPU Kernel 测试。

完成标志：

```text
Shape([4,2,3]) -> Gather(0) -> Unsqueeze -> Concat([6])
```

能得到 Int64 数据 `[4, 6]`。

### 阶段 B：增加双输入 `ReshapeObj`

1. 保留静态构造函数；
2. 新增动态构造函数；
3. 实现 `resolveReshapeSpec()`；
4. 实现 `resolveRuntimeShape()`；
5. 增加 `GraphHandlerObj::reshapeDynamic()` 和 pybind；
6. 先写一个手工调用 `resolveRuntimeShape()` 的 C++ 单测。

完成标志：Shape Tensor 数据为 `[4,6]` 时，输出 Tensor 元数据变为 `[4,6]`，非法 `[-1,-1]`、元素数不一致和越界 0 能明确报错。

### 阶段 C：接入 Runtime 两阶段执行

1. 反向收集动态 `Reshape` 的 Shape producer；
2. 在 Native CPU Runtime 中先执行白名单 Shape 子图；
3. 逐个解析动态 `Reshape`；
4. `shape_infer()` 更新后继；
5. `remallocForCurrentShape()`；
6. 从头执行完整图；
7. 跑五组连续 Shape C++ 测试。

### 阶段 D：接入 ONNX importer 和端到端测试

1. initializer shape 继续走静态接口；
2. 非 initializer shape 走 `reshape_dynamic`；
3. 明确拒绝未支持的 `Shape start/end` 和 `allowzero=1`；
4. 修正 `to_onnx()`；
5. 跑 Python 端到端测试；
6. 最后与 ONNX Runtime 对齐五组输入。

## 10. 建议的测试命令

根据本机已有构建配置调整 build 目录。典型命令：

```powershell
cmake -S . -B build -DBUILD_TEST=ON -DBUILD_TEST_CORE=ON
cmake --build build --config Release --target test_shape test_nativecpu_shape_tensor
ctest --test-dir build -C Release -R "shape|reshape" --output-on-failure
```

Python 测试：

```powershell
python -m pytest pyinfinitensor/tests/test_onnxstub.py -k "DynamicShapeTensor" -q
```

再跑已有回归：

```powershell
python -m pytest pyinfinitensor/tests/test_onnxstub.py -q
ctest --test-dir build -C Release --output-on-failure
```

## 11. 容易踩的坑

### 11.1 `Shape` 输出不是输入 dtype

无论输入是 `Float32/Float16/Int8`，ONNX `Shape` 输出都应为 `Int64`。

### 11.2 Shape Tensor 的 Shape 和数据不能混淆

```text
target.getDims() = [2]
target.data      = [batch, 6]
```

动态 `Reshape` 读取的是 `target.data`。

### 11.3 不能在普通 `shape_infer()` 中读取上次运行的 Blob

Blob 中的 `[batch,6]` 可能仍属于上一次输入。只有本次 Shape 子图刚算完才能解析。

### 11.4 二次分配后 Shape 中间值会失效

这是为什么最终数据阶段必须从头重新执行 Shape 子图，不能从 `Reshape` 后面直接继续。

### 11.5 `Gather` 必须在 Kernel 中检查 index

构图时动态 index 可能没有数据，只在 Operator 构造时检查不够。

### 11.6 `Concat` 的 Kernel dtype 分派不能只支持 Float32

Shape Tensor 是 Int64。只改 Operator Shape 推导而不改 Kernel 会在运行时失败。

### 11.7 `getWorkloadVector()` 也是执行链的一部分

CPU Runtime 在查 Kernel 后会创建性能键。`ShapeObj` 若不覆盖 workload/attr，即使 Kernel 已实现也会在执行前 halt。

### 11.8 不要破坏静态 Reshape

initializer 驱动的 ONNX `Reshape` 继续走现有 `reshape(data, output, Shape)`。新增路径只处理第二输入由运行时节点产生的情况。

### 11.9 用户当前 `onnx.py` 有未提交修改

当前工作区的 `pyinfinitensor/src/pyinfinitensor/onnx.py` 已有动态维度表示相关未提交修改。实现本文代码时应在现有内容上增量编辑，不要用 `git checkout`、整文件覆盖或复制旧版本，以免丢失前一任务的工作。

## 12. 第一版验收清单

- [ ] `ShapeObj` 输出 Tensor 的 dtype 为 `Int64`；
- [ ] Native CPU `Shape` 数值正确；
- [ ] Native CPU `Gather` 支持 Int64 data、Int32/Int64 indices 和负 index；
- [ ] Native CPU `Concat` 支持 Int64；
- [ ] Native CPU `Cast` 至少覆盖 Shape 子图需要的转换；
- [ ] `Squeeze/Unsqueeze::inferShape()` 多次调用结果稳定；
- [ ] 静态 `Reshape` 原测试不回归；
- [ ] 动态 `Reshape` 第二输入可以来自 `Concat` 输出；
- [ ] Shape 子图在最终内存规划之前运行；
- [ ] 动态输出 Tensor 的 Shape 和 Blob 字节数一致；
- [ ] 同一模型连续执行 `1 -> 2 -> 8 -> 3 -> 1`；
- [ ] 每次输出 Shape 都是 `[batch,6]`；
- [ ] 每次数值与 NumPy/ONNX Runtime 一致；
- [ ] 非法 shape spec 有清晰错误；
- [ ] 现有静态 ONNX 测试通过。

## 13. 后续扩展，不要混入第一版

第一版通过后再逐项扩展：

1. `Shape(start, end)`；
2. `Reshape allowzero=1`；
3. `Expand` 的运行时 Shape Tensor；
4. `ConstantOfShape`；
5. Shape 子图中的整数 `Add/Mul/Slice`；
6. IntelCPU/CUDA Shape Kernel；
7. Shape 子图常量折叠和部分求值；
8. 动态 rank；
9. CUDA Graph 按 Shape/地址变化重新捕获或缓存。

先把本文件描述的最小闭环做通，会比同时扩展这些功能更容易定位问题，也已经覆盖训练营通过标准要求的核心链路。
