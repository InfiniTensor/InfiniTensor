# 运行时 Shape Tensor 驱动动态 Reshape 实现指南

## 1. 本阶段要完成什么

本阶段建立下面这条完整的数据流：

```text
X: [batch, 2, 3]
 │
 ├──────────────────────────────────────────────┐
 │                                              │
 └─ Shape(X)                 data=[batch,2,3]    │
       └─ Gather(index=0)    data=batch          │
             └─ Unsqueeze   data=[batch]        │
                    │                            │
Constant([6]) ──────┴─ Concat   data=[batch,6]  │
                             │                   │
                             └──── Reshape(X, shape)
                                                   │
                                                   Y: [batch, 6]
```

同一个 ONNX 模型实例应当可以连续运行：

```text
batch=1 -> Y.shape=[1, 6]
batch=4 -> Y.shape=[4, 6]
batch=2 -> Y.shape=[2, 6]
```

这里真正新增的能力不是“给 `Reshape` 多传一个参数”，而是：

1. `ReshapeObj` 能把第二个输入保存为真正的 Tensor；
2. Runtime 能先执行 Shape Tensor 子图；
3. Runtime 读取 Shape Tensor 的数据，得到本次运行的目标 Shape；
4. 新 Shape 向后传播后，Graph 按新大小重新规划内存；
5. 最后从头执行完整计算图。

本指南假设你已经完成基础 Shape Tensor 算子：

- `Shape`；
- `Gather`；
- `Unsqueeze/Squeeze`；
- `Concat` 的 `Int64` 支持；
- `Cast` 的 Shape Tensor 常用类型支持。

第一版建议限定为：

- 仅支持 `Native CPU`；
- 动态的是 Shape Tensor 中的维度值，目标 Tensor 的 rank 固定；
- `Reshape` 先支持 ONNX 默认的 `allowzero=0`；
- Shape 子图白名单先支持 `Shape/Gather/Unsqueeze/Squeeze/Concat/Cast/Identity`；
- 保留现有静态 `Reshape(data, dims)`，不改变静态模型的行为；
- 暂不支持 CUDA Graph、固定大小 memory pool 和动态 `Expand/Resize/Slice`。

## 2. 为什么必须修改 Runtime

当前 CPU Runtime 的执行顺序可以简化成：

```text
dataMalloc()：按所有 Tensor 当前的 Shape 分配内存
                         ↓
run()：按拓扑序把所有 Kernel 执行一遍
```

但动态 `Reshape` 的目标 Shape 是普通 Tensor 数据。例如：

```text
targetShape.getDims() = [2]       // targetShape 自身是长度 2 的一维 Tensor
targetShape.data      = [4, 6]    // Reshape 真正需要的目标 Shape
```

`[4, 6]` 要等 `Shape/Gather/Concat` 的 Kernel 执行后才存在。若仍按旧流程，Runtime 在执行前不知道 `Y` 应分配多少字节。

因此新流程必须是：

```text
第一次分配：使用构图期占位 Shape
                  ↓
Shape 阶段：只执行目标 Shape 所依赖的小子图
                  ↓
读取 shapeTensor.data，解析目标 Shape
                  ↓
更新动态 Reshape 和后继 Tensor 的 Shape 元数据
                  ↓
第二次分配：按本次真实 Shape 重规划内存
                  ↓
完整执行：从头执行整张图
```

不能在完整执行到一半时临时重分配。整图重规划可能移动中间激活值；此前已经算出的中间结果可能随即失效。

## 3. 需要修改的文件

| 文件 | 修改目的 |
|---|---|
| `include/operators/reshape.h` | 为 `ReshapeObj` 增加双 Tensor 输入模式和运行时解析接口 |
| `src/operators/reshape.cc` | 实现 ONNX Reshape 规格解析、占位 Shape、运行时读取 Shape Tensor |
| `include/core/graph.h` | 增加沿用当前 allocator 模式重新规划内存的入口 |
| `src/core/graph.cc` | 实现 `remallocForCurrentShapes()` |
| `src/core/runtime.cc` | 收集并预执行 Shape 子图，解析动态 Reshape，再重分配 |
| `include/core/graph_handler.h` | 增加 `reshapeDynamic` 接口 |
| `src/core/graph_handler.cc` | 把 data 和 shapeTensor 一起传给 `ReshapeObj` |
| `src/ffi/ffi_infinitensor.cc` | 向 Python 暴露动态接口，并给 ONNX 导出提供模式查询 |
| `pyinfinitensor/src/pyinfinitensor/onnx.py` | 常量 shape 走静态路径，运行时 shape 走动态路径 |
| `test/operators/test_reshape.cc` | 测 Reshape 规格解析和双输入 Operator |
| `test/kernels/nativecpu/test_nativecpu_dynamic_reshape.cc` | 测 Runtime 两阶段执行和连续多次运行 |
| `pyinfinitensor/tests/test_onnxstub.py` | 测 ONNX Shape 子图驱动 Reshape 的端到端行为 |

现有 `src/kernels/cpu/reshape.cc` 通常不需要修改。`Reshape` 不改变元素排列，最终 Kernel 仍然只需把第一个输入的数据复制到输出：

```cpp
auto size = _op->getInputs()[0]->getBytes();
void *inptr = _op->getInputs(0)->getRawDataPtr<void *>();
void *outptr = _op->getOutput()->getRawDataPtr<void *>();
std::memcpy(outptr, inptr, size);
```

第二输入只决定输出的 Shape，不参与数据复制。

## 4. 第一步：让 `ReshapeObj` 支持两种模式

### 4.1 设计原则

保留旧构造函数：

```cpp
ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims);
```

新增动态构造函数：

```cpp
ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor,
           Tensor output, bool allowZero = false);
```

两种模式的区别如下：

| 模式 | `inputs` | Shape 规格来自哪里 | 何时能确定输出 Shape |
|---|---|---|---|
| 静态 | `{data}` | C++ 成员 `dims` | 构图/普通 `shape_infer()` |
| 动态 | `{data, shapeTensor}` | `inputs[1].data` | Shape 子图执行之后 |

动态模式下，普通 `inferShape()` 不能读取 `inputs[1].data`。构图时这块数据可能还没有分配；调用 `set_input()` 时，它也可能仍是上一次运行的旧值。

因此拆成两个职责：

```text
inferShape()          只做纯元数据传播
resolveRuntimeShape() 只在 Shape 子图刚执行完时读取 Tensor 数据
```

### 4.2 修改 `include/operators/reshape.h`

把 `ReshapeObj` 部分改成：

```cpp
class ReshapeObj : public OperatorObj {
    Shape dims;        // 静态模式的原始规格，例如 {0, -1}
    Shape outputShape; // 当前已经解析出的实际输出 Shape
    bool runtimeShape = false;
    bool allowZero = false;

  public:
    // 原有静态路径。
    ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims);

    // 新增运行时 Shape Tensor 路径。
    ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor,
               Tensor output, bool allowZero = false);

    OP_CLONE(ReshapeObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override {
        return static_cast<int>(inputs.size());
    }
    int numOutputs() const override { return 1; }

    bool isRuntimeShape() const { return runtimeShape; }
    bool getAllowZero() const { return allowZero; }

    Tensor getShapeTensor() const {
        return runtimeShape ? inputs.at(1) : nullptr;
    }

    // 返回本次解析是否改变了输出 Shape。
    bool resolveRuntimeShape();

    Shape getShape() const { return outputShape; }
    Shape getDims() const { return dims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
```

`numInputs()` 不能再固定返回 `1`，否则动态模式虽然在 `inputs` 中保存了两个 Tensor，框架却仍认为算子只有一个输入。

### 4.3 修改 `src/operators/reshape.cc`

文件头建议显式补齐：

```cpp
#include "operators/reshape.h"
#include "utils/operator_utils.h"
#include <algorithm>
#include <limits>
#include <numeric>
```

在 `namespace infini` 内增加两个私有辅助函数：

```cpp
namespace {

Shape resolveReshapeSpec(const Shape &inputShape, size_t inputSize,
                         const vector<int64_t> &spec, bool allowZero) {
    Shape result(spec.size());
    int inferAxis = -1;
    bool hasLiteralZero = false;
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

        if (value == 0) {
            if (allowZero) {
                hasLiteralZero = true;
            } else {
                IT_ASSERT(i < inputShape.size(),
                          "Reshape 0 dimension exceeds input rank");
                value = inputShape[i];
            }
        }

        IT_ASSERT(value <= std::numeric_limits<int>::max(),
                  "Reshape dimension exceeds InfiniTensor Shape range");
        result[i] = static_cast<int>(value);

        if (value == 0) {
            knownProduct = 0;
        } else if (knownProduct != 0) {
            const size_t dim = static_cast<size_t>(value);
            IT_ASSERT(knownProduct <=
                          std::numeric_limits<size_t>::max() / dim,
                      "Reshape element count overflow");
            knownProduct *= dim;
        }
    }

    if (inferAxis >= 0) {
        IT_ASSERT(!(allowZero && hasLiteralZero),
                  "Reshape allowzero=1 cannot combine 0 and -1");
        IT_ASSERT(knownProduct != 0,
                  "Reshape -1 with a zero known product is ambiguous");
        IT_ASSERT(inputSize % knownProduct == 0,
                  "Reshape -1 dimension cannot be inferred exactly");

        const size_t inferred = inputSize / knownProduct;
        IT_ASSERT(inferred <= static_cast<size_t>(
                                  std::numeric_limits<int>::max()),
                  "Inferred Reshape dimension is too large");
        result[inferAxis] = static_cast<int>(inferred);
    } else {
        IT_ASSERT(knownProduct == inputSize,
                  "Reshape input and output element counts differ");
    }

    return result;
}

vector<int64_t> toInt64(const Shape &shape) {
    return vector<int64_t>(shape.begin(), shape.end());
}

Shape makeRuntimePlaceholder(const Tensor &input, size_t outputRank) {
    // 占位 Shape 只在第一次 Shape 阶段之前使用。让它与输入元素数相同，
    // 比全部填 1 更安全。
    if (outputRank == input->getRank())
        return input->getDims();

    if (outputRank == 0) {
        IT_ASSERT(input->size() == 1,
                  "Scalar Reshape placeholder requires one input element");
        return {};
    }

    IT_ASSERT(input->size() <= static_cast<size_t>(
                                   std::numeric_limits<int>::max()),
              "Reshape placeholder dimension is too large");
    Shape placeholder(outputRank, 1);
    placeholder[0] = static_cast<int>(input->size());
    return placeholder;
}

} // namespace
```

然后替换 `ReshapeObj` 的相关实现；文件后面的 `FlattenObj` 和 `IdentityObj` 保留：

```cpp
ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}),
      dims(std::move(dims)), runtimeShape(false), allowZero(false) {
    IT_ASSERT(checkValid(graph));
}

ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor shapeTensor,
                       Tensor output, bool allowZero)
    : OperatorObj(OpType::Reshape, {input, shapeTensor}, {output}),
      outputShape(output ? output->getDims()
                         : makeRuntimePlaceholder(input,
                                                  shapeTensor->size())),
      runtimeShape(true), allowZero(allowZero) {
    IT_ASSERT(shapeTensor->getRank() == 1,
              "Reshape shape input must be a 1-D Tensor");
    IT_ASSERT(shapeTensor->getDType() == DataType::Int64,
              "ONNX Reshape shape input must be Int64");
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) {
    if (runtimeShape) {
        IT_ASSERT(inputs.size() == 2);
        // 不读取 inputs[1] 的数据。这里只传播最近一次已解析结果，
        // 第一次调用时传播构造函数建立的占位 Shape。
        return {{outputShape}};
    }

    IT_ASSERT(inputs.size() == 1);
    outputShape = resolveReshapeSpec(inputs[0]->getDims(), inputs[0]->size(),
                                     toInt64(dims), allowZero);
    return {{outputShape}};
}

bool ReshapeObj::resolveRuntimeShape() {
    IT_ASSERT(runtimeShape,
              "resolveRuntimeShape is only valid for dynamic Reshape");

    const Tensor shapeTensor = inputs.at(1);
    IT_ASSERT(shapeTensor->getRank() == 1,
              "Reshape shape input rank changed at runtime");
    IT_ASSERT(shapeTensor->size() == outputShape.size(),
              "Dynamic Reshape output rank cannot change in the first version");
    IT_ASSERT(shapeTensor->hasData(),
              "Runtime Reshape shape Tensor has no data");
    IT_ASSERT(shapeTensor->getDataBlob()->getBytes() ==
                  shapeTensor->getBytes(),
              "Runtime Reshape shape Tensor storage is invalid");

    const vector<int64_t> spec = shapeTensor->copyout<int64_t>();
    Shape resolved = resolveReshapeSpec(inputs[0]->getDims(),
                                        inputs[0]->size(), spec, allowZero);

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
```

### 4.4 这里几个容易写错的点

第一，动态模式的 `getOpAttrVector()` 不要加入当前 `outputShape`：

```cpp
// 错误思路：动态运行结果不是算子静态属性。
return {type.underlying(), outputShape...};
```

算子属性哈希应描述算子语义；运行时解析出的 Shape 会变化。它应该进入 `getWorkloadVector()`，不应该成为动态算子的静态属性。

第二，`0` 与 `-1` 的含义必须按 ONNX 规则处理：

```text
input Shape = [2, 3, 4]
spec = [0, -1]
allowzero = 0

0 复制 input[0]，得到 2
-1 根据总元素数推导，得到 12
output Shape = [2, 12]
```

第三，动态模式中：

```cpp
inputs[0] // 要被 reshape 的数据 Tensor
inputs[1] // 保存目标维度数值的 Shape Tensor
```

不要把 `inputs[1]->getDims()` 当成目标 Shape。若 `inputs[1]` 的数据为 `[4, 6]`：

```text
inputs[1]->getDims() = [2]
inputs[1].data       = [4, 6]
```

## 5. 第二步：让 Graph 能按原 allocator 模式重分配

### 5.1 修改 `include/core/graph.h`

在 `dataMalloc()` 后增加：

```cpp
void dataMalloc(bool useNaiveAllocator = false, size_t memPoolSize = 0);

// Shape 元数据改变后，沿用首次分配时锁定的 allocator 模式重新规划。
void remallocForCurrentShapes();
```

### 5.2 修改 `src/core/graph.cc`

在 `GraphObj::dataMalloc()` 后增加：

```cpp
void GraphObj::remallocForCurrentShapes() {
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

不能在 Runtime 中直接写：

```cpp
graph->dataMalloc();
```

因为用户可能第一次使用的是 naive allocator。现有 `lockAllocationMode()` 禁止中途把 naive allocator 切换为动态内存池。

现有分配器在重规划时会保留：

- weight Tensor；
- 没有 producer 的输入 Tensor 数据。

这很重要：用户通常在 `run()` 前已经把模型输入复制进 Tensor；Shape 阶段后的重分配不能丢掉它。

固定 memory pool 已经会拒绝布局变化。第一版保留这个限制，不要绕过检查。

## 6. 第三步：在 CPU Runtime 中增加 Shape 阶段

### 6.1 Shape 子图怎样识别

从每个动态 `Reshape` 的第二输入反向追踪 producer：

```text
Reshape.inputs[1]
       ↑
     Concat
     ↑    ↑
Unsqueeze Constant
   ↑
 Gather
   ↑
 Shape
```

第一版白名单：

```cpp
Shape, Gather, Unsqueeze, Squeeze, Concat, Cast, Identity
```

遇到 `Shape` 后停止向前追踪，因为 `Shape` 只读取输入 Tensor 的 Shape 元数据，不需要先算出输入 Tensor 的数值。

若继续追踪 `Shape` 的输入，可能把前面的卷积、MatMul 等整个数据图都错误地纳入 Shape 阶段。

### 6.2 修改 `src/core/runtime.cc`

补充 include：

```cpp
#include "operators/reshape.h"
#include <unordered_set>
```

在 `namespace infini` 内、`CpuRuntimeObj::run()` 前加入：

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

void collectShapeValueOps(
    const Tensor &tensor,
    std::unordered_set<OperatorObj *> &requiredShapeOps) {
    const Operator source = tensor->getSource();

    // initializer、Constant，或者调用者直接传入的 Shape Tensor。
    if (!source)
        return;

    IT_ASSERT(isShapeValueOp(source->getOpType()),
              std::string("Runtime Reshape shape input depends on unsupported ") +
                  "operator: " + source->getOpType().toString());

    if (!requiredShapeOps.insert(source.get()).second)
        return; // 已访问，避免共享子图被重复遍历。

    if (source->getOpType() == OpType::Shape)
        return;

    for (const Tensor &input : source->getInputs())
        collectShapeValueOps(input, requiredShapeOps);
}

bool prepareRuntimeShapes(const Graph &graph, Device device,
                          const RuntimeObj *runtime) {
    bool hasDynamicReshape = false;
    std::unordered_set<OperatorObj *> requiredShapeOps;

    IT_ASSERT(graph->topo_sort(), "Graph contains a cycle");

    // 第一遍：找到所有动态 Reshape，并收集它们依赖的 Shape 子图。
    for (const Operator &op : graph->getOperators()) {
        if (op->getOpType() != OpType::Reshape)
            continue;

        const auto reshape = as<ReshapeObj>(op);
        IT_ASSERT(reshape != nullptr);
        if (!reshape->isRuntimeShape())
            continue;

        hasDynamicReshape = true;
        collectShapeValueOps(reshape->getShapeTensor(), requiredShapeOps);
    }

    if (!hasDynamicReshape)
        return false;

    IT_ASSERT(device == Device::CPU,
              "Runtime Shape Tensor currently supports Native CPU only");

    const auto &kernelRegistry = KernelRegistry::getInstance();
    bool shapeChanged = false;

    // 第二遍按拓扑序执行。Shape producer 一定在使用它的 Reshape 前面。
    for (const Operator &op : graph->getOperators()) {
        if (requiredShapeOps.count(op.get())) {
            const KernelAttrs attrs{device,
                                    op->getOpType().underlying()};
            Kernel *kernel = kernelRegistry.getKernel(attrs);
            kernel->compute(op, runtime);
        }

        if (op->getOpType() != OpType::Reshape)
            continue;

        const auto reshape = as<ReshapeObj>(op);
        if (!reshape->isRuntimeShape())
            continue;

        shapeChanged = reshape->resolveRuntimeShape() || shapeChanged;

        // 让动态 Reshape 后面的普通算子立刻看到新 Shape。
        // 这也支持：DynamicReshape -> 普通算子 -> Shape -> DynamicReshape。
        graph->shape_infer();
    }

    if (shapeChanged) {
        graph->remallocForCurrentShapes();
        graph->validateMemory();
    }
    return true;
}

} // namespace
```

然后在 `CpuRuntimeObj::run()` 开头，把：

```cpp
graph->validateMemory();
```

改为：

```cpp
graph->validateMemory();
prepareRuntimeShapes(graph, device, this);
```

原有完整算子循环必须保留。一次 `run()` 最终会发生：

```text
prepareRuntimeShapes():
  Shape -> Gather -> Unsqueeze -> Concat
  -> resolveRuntimeShape
  -> shape_infer
  -> remallocForCurrentShapes

原有完整执行循环：
  Shape -> Gather -> Unsqueeze -> Concat
  -> Reshape memcpy
  -> 后续数据算子
```

Shape 中间 Tensor 的数据可能在重分配后失效，所以完整循环必须再算一次 Shape 子图。这是预期行为，不是重复计算 bug；Shape 子图通常很小。

### 6.3 为什么在每个动态 Reshape 后调用一次 `shape_infer()`

考虑更复杂的图：

```text
DynamicReshape-A -> Transpose -> Shape -> ... -> DynamicReshape-B
```

解析 A 后，`Transpose` 的输出元数据也要更新；后面的 `Shape` Kernel 才能读到正确 Shape。只在所有动态 Reshape 结束后统一调用 `shape_infer()`，B 依赖的 Shape 值可能已经按旧元数据计算。

第一版先保证正确性。以后若担心整图重复推导的开销，可以实现“从某个算子开始的增量 Shape 传播”。

## 7. 第四步：接通 GraphHandler 和 Python FFI

### 7.1 修改 `include/core/graph_handler.h`

在现有静态接口后增加：

```cpp
Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
Tensor reshapeDynamic(Tensor data, Tensor shapeTensor, Tensor reshaped,
                      bool allowZero = false);
```

### 7.2 修改 `src/core/graph_handler.cc`

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

### 7.3 修改 `src/ffi/ffi_infinitensor.cc`

在 GraphHandler 的 `.def(...)` 链中，紧跟静态 `reshape` 注册：

```cpp
.def("reshape", &Handler::reshape, policy::move)
.def("reshape_dynamic", &Handler::reshapeDynamic, policy::move)
```

为了让 `to_onnx()` 区分两种 Reshape，增加：

```cpp
static bool reshape_is_dynamic_of(Operator op) {
    IT_ASSERT(op->getOpType() == OpType::Reshape);
    return dynamic_cast<const ReshapeObj *>(op.get())->isRuntimeShape();
}
```

在已有 `.FUNCTION(...)` 注册链中增加：

```cpp
.FUNCTION(reshape_is_dynamic_of)
```

## 8. 第五步：修改 ONNX importer

### 8.1 为什么当前代码无法导入动态 Reshape

当前分支使用：

```python
shape = _parse_static_input(data, node, 1, required=True)
```

`data` 只保存 initializer 和 `Constant` 的 TensorProto。若第二输入来自 `Concat`，它不在 `data` 中，于是 importer 会直接报“必须是常量”。

新的分流规则是：

```text
node.input[1] 在 data 中     -> 原有静态 reshape
node.input[1] 不在 data 中   -> 新增动态 reshape_dynamic
```

### 8.2 为动态输出准备构图期占位信息

在创建 `tensors` 和 `data` 附近增加：

```python
tensors: Dict[str, backend.Tensor] = dict()
data: Dict[str, TensorProto] = dict()

declared_value_info = {
    value.name: value
    for value in list(model.graph.value_info) + list(model.graph.output)
}
```

ONNX 若声明：

```text
y: [batch, 6]
```

当前的动态维度表示会把它临时物化为：

```text
[1, 6]
```

这个占位 Shape 能帮助后续 `MatMul/Transpose` 等算子在构图时通过 rank 和固定维度检查。若 ONNX 没有相应 `value_info`，C++ 构造函数会退回到元素数相等的占位 Shape。

### 8.3 替换 `Reshape` 导入分支

把当前 `elif node.op_type == "Reshape"` 分支替换为：

```python
elif node.op_type == "Reshape":
    if not _has_input(node, 1):
        raise ValueError("Reshape requires a shape input")

    attributes = _parse_attribute(node, {"allowzero": 0})
    if attributes["allowzero"] != 0:
        raise NotImplementedError(
            "Reshape allowzero=1 is not supported in the first version"
        )

    shape_name = node.input[1]
    if shape_name in data:
        # initializer/Constant：保持原有静态路径。
        tensors[node.output[0]] = self.handler.reshape(
            tensors[node.input[0]],
            tensors.get(node.output[0]),
            _parse_data(data[shape_name]),
        )
    else:
        # Shape/Gather/Concat 等运行时子图。
        output = tensors.get(node.output[0])
        declared = declared_value_info.get(node.output[0])

        if output is None and declared is not None:
            tensor_type = declared.type.tensor_type
            output = self.handler.tensor(
                _materialize_shape(_parse_shape_spec(tensor_type.shape)),
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

这里传入的最后一个 `False` 对应 `allowZero`。

不要对动态路径调用 `_parse_data()`，因为 `shape_name` 此时不是 TensorProto，而是本图中一个真正的运行时 Tensor。

### 8.4 修改 ONNX 导出

当前 `to_onnx()` 会给所有 `Reshape` 无条件追加一个静态 shape initializer。动态 `Reshape` 本来已经有两个输入，不能再追加第三个。

把导出分支改为：

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

    # 动态模式的 inputs 已经是 [data, shapeTensor]。
    ctx.push_node(make_node(ty.name, inputs, outputs, name))
```

第一版拒绝了 `allowzero=1`，所以这里不用导出 `allowzero` 属性。

## 9. `set_input()` 与动态 Reshape 的关系

你现有的 Python 流程大致是：

```python
stub.set_input([[batch, 2, 3]])
stub.inputs["x"].copyin_numpy(x)
stub.run()
```

`set_input()` 当前会：

```text
修改输入 Tensor Shape
       ↓
普通 shape_infer()
       ↓
init()/data_malloc()
```

这一步仍应保留。它先让静态 Shape 能传播，并为 Shape 子图的输出准备可写缓冲区。

此时动态 `Reshape` 可能还是上一次解析的 Shape 或第一次构图的占位 Shape。真正的本次目标 Shape 会在随后 `run()` 的 `prepareRuntimeShapes()` 中确定，并在必要时触发第二次分配。

因此用户侧调用顺序不要改成先 `run()` 再复制输入；正确顺序仍是先 `set_input()`、再 `copyin`、最后 `run()`。

## 10. 测试代码

### 10.1 Operator 层：双输入 Reshape 解析

在 `test/operators/test_reshape.cc` 增加：

```cpp
TEST(Reshape, RuntimeShapeTensor) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor input = g->addTensor({2, 3, 4}, DataType::Float32);
    Tensor shape = g->addTensor({2}, DataType::Int64);
    auto op = g->addOp<ReshapeObj>(input, shape, nullptr, false);

    g->dataMalloc();
    shape->copyin<int64_t>({0, -1});

    EXPECT_TRUE(op->isRuntimeShape());
    EXPECT_TRUE(op->resolveRuntimeShape());
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 12}));
}
```

这个测试只验证：

- 第二输入被保存为 Shape Tensor；
- `0` 能复制对应输入维；
- `-1` 能根据元素数推导；
- 输出 Tensor 的 Shape 元数据被更新。

### 10.2 Runtime 层：直接由运行时 Shape Tensor 驱动

新增 `test/kernels/nativecpu/test_nativecpu_dynamic_reshape.cc`：

```cpp
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/reshape.h"
#include "test.h"

namespace infini {

TEST(DynamicReshape, RuntimeShapeInput) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    Tensor input = g->addTensor({2, 3}, DataType::Float32);
    Tensor shape = g->addTensor({2}, DataType::Int64);
    input->setInput();
    shape->setInput();

    auto op = g->addOp<ReshapeObj>(input, shape, nullptr, false);
    Tensor output = op->getOutput();
    output->setOutput();

    g->dataMalloc();
    input->copyin<float>({0, 1, 2, 3, 4, 5});
    shape->copyin<int64_t>({3, 2});

    runtime->run(g);
    EXPECT_EQ(output->getDims(), (Shape{3, 2}));
    EXPECT_TRUE(output->equalData(vector<float>{0, 1, 2, 3, 4, 5}));

    // 同一个图只改变 Shape Tensor 的数据，再运行一次。
    shape->copyin<int64_t>({1, 6});
    runtime->run(g);
    EXPECT_EQ(output->getDims(), (Shape{1, 6}));
    EXPECT_TRUE(output->equalData(vector<float>{0, 1, 2, 3, 4, 5}));
}

} // namespace infini
```

这个测试会真正走：

```text
读取运行时 Shape Tensor -> 更新 output Shape -> 重规划内存 -> Reshape memcpy
```

### 10.3 Python/ONNX 端到端测试

在 `pyinfinitensor/tests/test_onnxstub.py` 增加：

```python
class TestDynamicReshape(unittest.TestCase):
    def test_shape_subgraph_drives_reshape_across_runs(self):
        x_info = value_info("x", ["batch", 2, 3])
        y_info = value_info("y", ["batch", 6])

        index = initializer("index", np.array(0, dtype=np.int64))
        axes = initializer("axes", np.array([0], dtype=np.int64))
        six = initializer("six", np.array([6], dtype=np.int64))

        nodes = [
            helper.make_node("Shape", ["x"], ["x_shape"]),
            helper.make_node(
                "Gather", ["x_shape", "index"], ["batch_scalar"], axis=0
            ),
            helper.make_node(
                "Unsqueeze", ["batch_scalar", "axes"], ["batch_vec"]
            ),
            helper.make_node(
                "Concat", ["batch_vec", "six"], ["target_shape"], axis=0
            ),
            helper.make_node("Reshape", ["x", "target_shape"], ["y"]),
        ]

        model = make_model(
            nodes,
            [x_info],
            [y_info],
            [index, axes, six],
        )
        stub = import_model(model)

        for batch in (1, 4, 2, 8, 1):
            with self.subTest(batch=batch):
                x = np.arange(batch * 6, dtype=np.float32).reshape(
                    batch, 2, 3
                )
                stub.set_input([[batch, 2, 3]])
                stub.inputs["x"].copyin_numpy(x)
                stub.run()

                self.assertEqual(stub.getShape("y"), [batch, 6])
                actual = np.asarray(
                    stub.outputs["y"].copyout_float(), dtype=np.float32
                ).reshape(batch, 6)
                np.testing.assert_array_equal(actual, x.reshape(batch, 6))
```

再补一个静态回归测试，确认 initializer shape 仍走旧路径：

```python
def test_static_reshape_still_uses_initializer(self):
    shape = initializer("shape", np.array([3, 2], dtype=np.int64))
    model = make_model(
        [helper.make_node("Reshape", ["x", "shape"], ["y"])],
        [value_info("x", [2, 3])],
        [value_info("y", [3, 2])],
        [shape],
    )

    stub = import_model(model)
    self.assertEqual(len(stub.handler.operators()[0].inputs()), 1)
```

若 Python 绑定的 `Operator` 没有暴露 `inputs()`，不要为了这一条断言额外扩大 FFI；改成导出 ONNX 后检查 Reshape 节点恰好有两个输入即可。

### 10.4 导出回归测试

动态模型导出后，应检查：

```python
exported = stub.to_onnx("dynamic_reshape")
checker.check_model(exported)
reshape_node = next(
    node for node in exported.graph.node if node.op_type == "Reshape"
)
self.assertEqual(len(reshape_node.input), 2)
```

若这里得到 3，说明 `to_onnx()` 又给动态 Reshape 追加了静态 initializer。

## 11. 推荐实现和验证顺序

### 阶段 A：先完成双输入 Operator

修改：

```text
include/operators/reshape.h
src/operators/reshape.cc
test/operators/test_reshape.cc
```

验收：

- 静态 `Reshape` 原测试通过；
- 动态构造函数能创建两个输入；
- `resolveRuntimeShape()` 正确处理正常维度、`0` 和 `-1`；
- 非 Int64、非一维 shape input 被拒绝。

### 阶段 B：加入 Runtime 两阶段执行

修改：

```text
include/core/graph.h
src/core/graph.cc
src/core/runtime.cc
test/kernels/nativecpu/test_nativecpu_dynamic_reshape.cc
```

验收：

- 直接传入 shape Tensor 可以连续改变 `{3,2}`、`{1,6}`；
- 输入数据在重分配后没有丢失；
- 输出 Tensor 的 Blob 大小与 `getBytes()` 一致；
- 静态 Reshape 不触发 Shape 预执行。

### 阶段 C：接通前端

修改：

```text
include/core/graph_handler.h
src/core/graph_handler.cc
src/ffi/ffi_infinitensor.cc
pyinfinitensor/src/pyinfinitensor/onnx.py
```

验收：

- initializer/Constant shape 仍走静态路径；
- `Shape -> Gather -> Unsqueeze -> Concat -> Reshape` 成功导入；
- 一个 stub 连续执行多个 batch；
- 动态模型再次导出时，Reshape 仍只有两个输入。

## 12. 建议运行的测试命令

下面命令本身不会修改源码；它们只编译或运行测试。构建目录请替换成你实际使用的目录。

C++ 定向测试：

```bash
ctest --test-dir build -R "test_reshape|test_nativecpu_dynamic_reshape" --output-on-failure
```

若测试文件是新加入的，先重新 configure，确保 CMake 的 glob 收集到它：

```bash
cmake -S . -B build -DBUILD_TEST=ON -DBUILD_TEST_CORE=ON
cmake --build build --target test_reshape test_nativecpu_dynamic_reshape -j
```

Python 定向测试：

```bash
python -m unittest pyinfinitensor.tests.test_onnxstub.TestDynamicReshape -v
```

Python importer 回归：

```bash
python -m unittest pyinfinitensor.tests.test_onnxstub -v
```

最后再运行全部 Native CPU/Core 测试，避免 Runtime 修改影响其他算子：

```bash
ctest --test-dir build --output-on-failure
```

## 13. 常见错误与定位方法

### 13.1 importer 仍报 shape 必须是常量

检查 `Reshape` 分支是否还无条件调用：

```python
_parse_static_input(data, node, 1, required=True)
```

动态分支只应根据 `shape_name in data` 判断是否为常量。

### 13.2 `ReshapeObj::checkValid()` 在构图期失败

通常是动态 `inferShape()` 试图读取 Shape Tensor 数据，或者输出占位 Shape 与预创建 output 不一致。

动态构造函数必须先设置：

```cpp
outputShape = output ? output->getDims() : makeRuntimePlaceholder(...);
```

然后才能调用 `checkValid(graph)`。

### 13.3 Shape Tensor 数值正确，但输出越界或 Blob 大小不匹配

说明只更新了：

```cpp
output->setShape(...)
```

却没有在完整数据 Kernel 执行前调用：

```cpp
graph->remallocForCurrentShapes();
```

Shape 元数据变化不会自动扩大已经分配的内存。

### 13.4 第二次运行结果仍使用第一次的 batch

检查每次 `run()` 是否都会执行 `prepareRuntimeShapes()`。不要只在构图或 `init()` 时解析一次 Shape Tensor。

### 13.5 重分配后输入变成零或随机值

检查输入 Tensor 是否被标记为 input：

```cpp
input->setInput();
```

并确认没有在完整执行到一半才重分配。重分配必须发生在任何数据算子执行之前。

### 13.6 Shape 子图找不到 Kernel

根据报错的 OpType 检查：

- 它是否应加入 Shape 子图白名单；
- Native CPU 是否已经有该 OpType 的 Kernel；
- dtype 是否覆盖 `Int64`；
- 新增 `.cc` 后是否重新运行过 CMake configure。

不要为了让报错消失就把所有算子加入白名单。白名单中的算子会在完整图之前额外执行一次，必须确认它只依赖已知元数据或 Shape Tensor 数据。

### 13.7 `getOpPerfKey()` 在 Shape 阶段前拿到旧 Shape

完整执行循环是在 `prepareRuntimeShapes()` 之后创建性能键的，因此应看到本次真实输出 Shape。若你把准备阶段放在原循环内部，就会出现旧 key、旧内存与新 Shape 混用。

## 14. “Reshape 等”后续怎样扩展

完成 `Reshape` 后，两阶段框架已经建立。其他动态 Shape 算子的接入方式可以统一成四步：

```text
1. Operator 保存 Shape Tensor 输入
2. inferShape() 不读取运行时数据
3. 增加 resolveRuntimeShape()，在 Shape 阶段更新输出元数据
4. Runtime 收集该算子的 Shape 输入，并在重分配前调用解析接口
```

例如动态 `Expand`：

```text
Expand(data, targetShapeTensor)
```

与 `Reshape` 共用 Shape 子图预执行和重分配机制，但它的完整数据 Kernel 不是 memcpy，需要按广播规则真正写输出。

动态 `Resize`：

```text
Resize(data, sizesTensor/scalesTensor)
```

需要在 Shape 阶段读取 `sizes/scales`，更新输出 Shape；完整阶段仍执行 Resize Kernel。

当第二个动态算子真正接入时，再把 Runtime 中的 `ReshapeObj` 特判抽象为统一接口，例如：

```cpp
class RuntimeShapeOp {
  public:
    virtual ~RuntimeShapeOp() = default;
    virtual TensorVec getRuntimeShapeInputs() const = 0;
    virtual bool resolveRuntimeShape() = 0;
};
```

第一版只有 `Reshape` 时不建议先引入这层抽象。先用端到端测试把执行时序和内存生命周期验证正确，再做公共接口，调试范围会小很多。

## 15. 最终验收清单

- [ ] 现有静态 `Reshape` 构造函数和 importer 路径仍保留；
- [ ] 动态 `ReshapeObj` 的 `inputs` 是 `{data, shapeTensor}`；
- [ ] shape input 必须是一维 `Int64` Tensor；
- [ ] 普通 `inferShape()` 不读取 Tensor 数据；
- [ ] `resolveRuntimeShape()` 正确处理正整数、`0` 和单个 `-1`；
- [ ] 非法负数、多个 `-1`、元素数不等能明确失败；
- [ ] Shape 子图只预执行白名单算子，并在 `Shape` 处停止反向追踪；
- [ ] 动态 Shape 在任何完整数据算子执行前完成解析；
- [ ] Shape 变化后沿用原 allocator 模式重新规划内存；
- [ ] 重分配后用户输入和 initializer 数据仍正确；
- [ ] 同一个模型实例可连续执行多个 batch；
- [ ] ONNX 动态 Reshape 导出后仍恰好有两个输入；
- [ ] Operator、Native CPU、ONNX importer 的定向测试全部通过；
- [ ] 全部 Core/Native CPU 回归测试通过。

做到这些，才算真正完成“运行时 Shape Tensor 驱动的 `Reshape`”：不仅导入成功，而且 Shape 值、元数据传播、内存大小和最终数值在连续多次运行中都一致。
