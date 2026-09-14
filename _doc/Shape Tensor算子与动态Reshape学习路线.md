# Shape Tensor 算子与动态 Reshape 学习路线

## 1. 学习目标

这份文档服务于以下两个开发任务：

1. 支持项目所需的基础 Shape Tensor 相关算子；
2. 支持由运行时 Shape Tensor 驱动的 `Reshape` 等动态 Shape 算子。

这一阶段最关键的概念是：

> Tensor 的 Shape 元数据和 Shape Tensor 是两种不同的信息。

例如输入 Tensor 的实际 Shape 为 `[4, 2, 3]`：

```text
Tensor X
  X.getDims() = [4, 2, 3]
  X.data      = 大量模型输入数据

Shape(X) 的输出 S
  S.getDims() = [3]
  S.dtype     = Int64
  S.data      = [4, 2, 3]
```

`X.getDims()` 是 C++ Tensor 的元数据；`S.data` 是一段正常的 Tensor 数据。动态 `Reshape` 需要读取的是后者。

## 2. 用一个完整 Shape 子图贯穿阅读

阅读代码时始终带着下面这个模型：

```text
X: ["batch", 2, 3]
│
├─ Shape(X)
│    S.shape = [3]
│    S.data  = [batch, 2, 3]
│
├─ Gather(S, index=0)
│    G.shape = []
│    G.data  = batch
│
├─ Unsqueeze(G, axes=[0])
│    U.shape = [1]
│    U.data  = [batch]
│
├─ Constant([6])
│
├─ Concat(U, [6])
│    RS.shape = [2]
│    RS.data  = [batch, 6]
│
└─ Reshape(X, RS)
     Y.shape = [batch, 6]
```

当 `batch=4` 时，运行时的值流为：

```text
X.getDims()                = [4, 2, 3]
Shape(X)                   = [4, 2, 3]
Gather([4, 2, 3], 0)       = 4
Unsqueeze(4, axes=[0])     = [4]
Concat([4], [6])           = [4, 6]
Reshape(X, [4, 6]).getDims = [4, 6]
```

其中需要严格区分：

```text
X.getDims()  = [4, 2, 3]  # X 的 Shape 元数据
RS.getDims() = [2]        # Shape Tensor 自己的 Shape 元数据
RS.data      = [4, 6]     # 动态 Reshape 需要读取的数据
```

## 3. 第一组：Tensor 的元数据和实际数据

### 阅读文件

```text
include/core/tensor.h
include/core/tensor_base.h
src/core/tensor.cc
include/core/blob.h
src/core/blob.cc
```

### 重点成员

```cpp
Shape shape;     // Tensor 的 Shape 元数据
size_t _size;    // Tensor 元素数量
DataType dtype;  // 元素类型
Blob data;       // Tensor 的实际数据存储
```

### 重点函数

```cpp
getDims()
setShape()
size()
getBytes()
dataMalloc()
copyin()
copyout()
getRawDataPtr()
```

带着 Shape Tensor `RS` 理解：

```text
RS.getDims()  → [2]
RS.getDType() → Int64
RS.copyout()  → [4, 6]
RS.getBytes() → 2 × sizeof(int64_t)
```

### 阅读完成标准

- 能解释 `setShape()` 修改的是元数据，而 Kernel 写入的是 `Blob data`；
- 能解释 Shape Tensor 为什么也需要正常的 dtype、Blob 和内存分配；
- 能解释动态 `Reshape` 为什么要从 Shape Tensor 的 Blob 中读取值。

## 4. 第二组：Operator 的创建和 Shape 推导

### 阅读文件

```text
include/core/operator.h
src/core/operator.cc
```

### 重点函数

```cpp
inferShape()
inferDataType()
checkValid()
numInputs()
numOutputs()
getInputs()
getOutputs()
getWorkloadVector()
getOpAttrVector()
```

`OperatorObj::checkValid()` 的核心流程是：

```text
调用 inferShape()
        ↓
如果正在创建 Operator
        ↓
根据推导出的 Shape 和 dtype 创建输出 Tensor
```

这会引出动态 `Reshape` 的构图问题：

```text
构图时 Shape Tensor 还没有执行
        ↓
Shape Tensor 里还没有 [batch, 6]
        ↓
ReshapeObj 构造时无法获得最终输出 Shape
```

因此动态 `Reshape` 最终需要区分：

```text
构图阶段的初始/占位 Shape
运行前解析出的 resolved Shape
```

### 阅读完成标准

- 能解释 Operator 和 Kernel 的职责差异；
- 能解释为什么大部分 Operator 构造时必须能推导输出 Shape；
- 能指出这个假设为什么不完全适用于 Tensor 驱动的动态 `Reshape`。

## 5. 第三组：Kernel 注册和 Runtime 执行

### 阅读文件

```text
include/core/kernel.h
include/core/runtime.h
src/core/runtime.cc
```

### 重点代码

```cpp
REGISTER_KERNEL(Device::CPU, OpType::..., KernelClass, "kernel_name")
```

CPU Runtime 的核心执行方式：

```cpp
for (auto &op : graph->getOperators()) {
    Kernel *kernel = kernelRegistry.getKernel(...);
    kernel->compute(op, this);
}
```

职责关系：

```text
Operator
  负责图结构、输入输出、Shape 推导和 dtype 推导

Kernel
  负责读取输入 Blob，计算并写入输出 Blob

Runtime
  负责按拓扑顺序查找并执行 Kernel
```

以 `Shape` 为例：

```text
ShapeObj
  输出 Tensor Shape = [输入 rank]
  输出 Tensor dtype = Int64

ShapeCpuKernel
  读取 input->getDims()
  将维度值写入 output->data
```

### 阅读完成标准

- 能找到一个 Operator 对应的 CPU Kernel 注册位置；
- 能解释 Runtime 如何从 `OpType` 找到 Kernel；
- 能写出一个读取输入 Tensor、写入输出 Tensor 的简单 CPU Kernel。

## 6. 第四组：逐个阅读 Shape Tensor 算子

## 6.1 Shape

### 阅读文件

```text
include/operators/unary.h
src/operators/unary.cc
src/kernels/cpu/unary.cc
src/core/graph_handler.cc
pyinfinitensor/src/pyinfinitensor/onnx.py
```

重点搜索：

```text
ShapeObj
GraphHandlerObj::shape
node.op_type == "Shape"
```

当前已有：

- `ShapeObj` 类；
- 输出 Tensor Shape 为 `[input_rank]` 的初步推导；
- GraphHandler 接口；
- ONNX importer 分支。

当前缺口：

- `ShapeObj` 没有覆盖 `inferDataType()`；
- ONNX `Shape` 输出必须是 Int64，而当前会继承输入 dtype；
- Native CPU 没有 `Shape` Kernel；
- 缺少自己的 workload/attribute 实现；
- importer 没有处理或拒绝 `start/end` 属性。

必须学习的 ONNX 语义：

- 输出是一维 Int64 Tensor；
- 输出数据是输入 Tensor 的实际维度；
- 较新 opset 支持 `start/end` 对 Shape 进行切片；
-负 `start/end` 从尾部计数，并存在边界截断规则。

官方文档：<https://onnx.ai/onnx/operators/onnx__Shape.html>

第一版可以只支持完整 Shape。如果不支持 `start/end`，必须在 importer 中明确拒绝非默认属性，不能静默忽略。

## 6.2 Gather

### 阅读文件

```text
include/operators/gather.h
src/operators/gather.cc
src/kernels/cuda/gather.cc
src/kernels/intelcpu/gather.cc
pyinfinitensor/src/pyinfinitensor/onnx.py
```

当前已有：

- `GatherObj`；
- axis 处理；
-输出 Shape 推导；
- indices dtype 检查；
- ONNX importer 分支；
- CUDA、Intel CPU 等后端 Kernel。

当前缺口：

- Native CPU 没有 Gather Kernel；
-需要重点检查负 index、负 axis 和边界错误；
- Shape Tensor 链要求 Gather 能处理 Int64 data。

输出 Shape 公式：

```text
data rank    = r
indices rank = q
output rank  = q + r - 1
```

简单 Shape 链中：

```text
data.shape    = [3]
data.data     = [4, 2, 3]
indices.shape = []
indices.data  = 0
output.shape  = []
output.data   = 4
```

不要把 `Gather` 和 `GatherElements` 混淆。

官方文档：<https://onnx.ai/onnx/operators/onnx__Gather.html>

## 6.3 Unsqueeze 和 Squeeze

### 阅读文件

```text
include/operators/unsqueeze.h
src/operators/unsqueeze.cc
include/operators/squeeze.h
src/operators/squeeze.cc
src/kernels/cpu/reshape.cc
pyinfinitensor/src/pyinfinitensor/onnx.py
```

这两个算子不会改变元素内容，只改变 Tensor Shape，因此 Native CPU Kernel 可以直接复制数据。

需要掌握：

- 输入输出元素数量相同；
- axes 可以为负；
- axes 不能重复；
- `Squeeze` 指定轴的维度必须等于 1；
-旧 opset 中 axes 是属性；
-新 opset 中 axes 是第二个输入。

当前 importer 要求 axes 是常量。基础 Shape 链中动态的是维度值，不是 axes，因此这个限制可以暂时保留。

官方文档：

- <https://onnx.ai/onnx/operators/onnx__Unsqueeze.html>
- <https://onnx.ai/onnx/operators/onnx__Squeeze.html>

## 6.4 Concat

### 阅读文件

```text
include/operators/concat.h
src/operators/concat.cc
src/kernels/cpu/concat.cc
pyinfinitensor/src/pyinfinitensor/onnx.py
```

Operator 的 Shape 推导已经存在。重点阅读 Native CPU Kernel 的 dtype switch。

当前只覆盖：

```text
Float32
UInt32
```

Shape Tensor 通常为 Int64，因此至少需要增加：

```text
Int64
```

最好将模板分发扩展到项目支持的普通数值类型，而不是只为一个测试硬编码。

需要掌握：

-除拼接轴外，所有输入 Shape 必须一致；
- axis 允许为负；
-输出 dtype 与输入 dtype 相同；
- Shape Tensor 场景通常是沿 axis 0 拼接若干一维 Int64 Tensor。

官方文档：<https://onnx.ai/onnx/operators/onnx__Concat.html>

## 6.5 Cast

### 阅读文件

```text
include/operators/unary.h
src/operators/unary.cc
src/core/graph_handler.cc
src/kernels/cuda/unary.cc
src/kernels/cuda/unary.cu
pyinfinitensor/src/pyinfinitensor/onnx.py
```

重点搜索：

```text
CastType
CastObj
inferCastType
CastCuda
node.op_type == "Cast"
```

当前已有：

- `CastType` 枚举；
-输出 Shape 和 dtype 推导；
- ONNX `to` 到内部 CastType 的转换；
- CUDA Cast Kernel。

当前缺口：

- Native CPU 没有 Cast Kernel；
-需要确认项目实际 Shape 子图要求哪些转换组合。

基础范围建议覆盖：

```text
Int64 → Int32
Int32 → Int64
Int64 → Float32
Float32 → Int64
```

Cast 不改变 Shape，只逐元素改变 dtype。

官方文档：<https://onnx.ai/onnx/operators/onnx__Cast.html>

## 7. 第五组：当前静态 Reshape

### 阅读文件

```text
include/operators/reshape.h
src/operators/reshape.cc
src/kernels/cpu/reshape.cc
src/core/graph_handler.cc
pyinfinitensor/src/pyinfinitensor/onnx.py
test/operators/test_reshape.cc
```

当前结构的核心是：

```cpp
class ReshapeObj : public OperatorObj {
    Shape dims;
    Shape outputShape;

    int numInputs() const override { return 1; }
};
```

当前目标形状是 C++ Operator 属性：

```text
Reshape(data, dims属性)
```

而不是 ONNX 标准形式：

```text
Reshape(data, shapeTensor)
```

当前 ONNX importer 又调用：

```python
shape = _parse_static_input(data, node, 1, required=True)
```

所以只支持第二输入是 initializer 的 Reshape。

必须学习的 ONNX Reshape 语义：

- 第一个输入是 data；
-第二个输入是 Int64 Shape Tensor；
-最多一个 `-1`；
-默认情况下，`0` 表示复制输入对应维度；
- `allowzero=1` 时，`0` 表示真正的零维；
- `allowzero=1` 时，目标 Shape 不能同时包含 `0` 和 `-1`；
-输入输出元素总数必须一致；
-空 Shape Tensor 表示输出标量。

官方文档：<https://onnx.ai/onnx/operators/onnx__Reshape.html>

阅读完后，需要能解释动态版本为什么至少需要：

```cpp
inputs = {data, shapeTensor};
```

以及为什么建议增加显式状态或方法：

```cpp
bool hasDynamicShapeInput() const;
void refreshResolvedShape();
Shape getResolvedShape() const;
```

普通 `inferShape()` 不应在任意时刻读取 Shape Tensor，因为 Tensor 中可能仍然是上一次运行的旧值。Shape Tensor 必须在明确的 Shape 计算阶段完成后才能读取。

## 8. 第六组：为什么必须分阶段执行

### 阅读文件

```text
include/core/graph.h
src/core/graph.cc
include/core/runtime.h
src/core/runtime.cc
```

重点函数：

```cpp
GraphObj::topo_sort()
GraphObj::shape_infer()
GraphObj::dataMalloc()
CpuRuntimeObj::run()
```

当前主要执行顺序是：

```text
输入 Tensor.setShape()
        ↓
Graph.shape_infer()
        ↓
Graph.dataMalloc()
        ↓
Runtime.run(graph)
```

动态 Reshape 会形成如下依赖：

```text
给 Reshape 输出分配内存
        ↓ 需要
Reshape 输出 Shape
        ↓ 需要
Shape Tensor 中的目标维度值
        ↓ 需要
先执行 Shape 子图 Kernel
        ↓ 需要
Shape Tensor 自己已经有内存
```

因此最终流程需要变成：

```text
更新输入实际 Shape
        ↓
按当前/占位 Shape 分配基础内存
        ↓
执行 Shape 子图
        ↓
Dynamic Reshape 读取 Shape Tensor
        ↓
刷新 Reshape resolved Shape
        ↓
Graph.shape_infer() 向后传播
        ↓
Graph.dataMalloc() 重新规划
        ↓
执行完整数据计算图
```

这里需要掌握两个图算法概念。

### 8.1 拓扑排序

必须保证：

```text
Shape → Gather → Unsqueeze → Concat
```

按依赖顺序执行，不能先执行 Concat。

### 8.2 反向依赖闭包

从每个动态 `Reshape` 的第二输入开始，沿 Tensor 的 `source` 向前查找所有生产者：

```text
Reshape.shapeTensor
        ↓ source
Concat
        ↓ inputs/source
Unsqueeze、Constant
        ↓
Gather
        ↓
Shape
```

第一版可以将允许出现在闭包中的 Operator 限制为：

```text
Shape
Gather
Unsqueeze
Squeeze
Concat
Cast
Identity
```

如果闭包中出现未支持的 Operator，应报告明确错误，不能继续使用旧 Shape。

## 9. 第七组：Python/C++ 接口接线

### 阅读文件

```text
include/core/graph_handler.h
src/core/graph_handler.cc
src/ffi/ffi_infinitensor.cc
pyinfinitensor/src/pyinfinitensor/onnx.py
```

重点追踪：

```text
onnx.py
    ↓ self.handler.shape(...)
Python binding
    ↓
GraphHandlerObj::shape()
    ↓
GraphObj::addOp<ShapeObj>()
```

动态 Reshape 可能需要新增或扩展接口，例如：

```cpp
// 现有静态接口继续保留
reshape(data, output, staticShape);

// 新增 Tensor 驱动接口
reshape(data, shapeTensor, output, initialShape, allowZero);
```

最终还需要通过 pybind 暴露给 `onnx.py`。

不要在 FFI 中实现 Shape 子图计算。FFI 只负责 Python/C++ 参数桥接。

## 10. 第八组：OnnxStub 的构图与初始化时机

### 阅读位置

```text
pyinfinitensor/src/pyinfinitensor/onnx.py
```

重点搜索：

```python
node.op_type == "Shape"
node.op_type == "Gather"
node.op_type == "Squeeze"
node.op_type == "Unsqueeze"
node.op_type == "Concat"
node.op_type == "Cast"
node.op_type == "Reshape"
def init
def set_input
def run
```

需要理解四件事：

1. ONNX Node 如何创建 C++ Operator；
2. initializer 什么时候复制到 Tensor；
3.输入 Shape 什么时候改变；
4. Shape 子图预执行应该插入哪个阶段。

当前初始化过程包含：

```python
def init(self):
    self.handler.data_malloc(...)
    self._copy_initializers()
```

Shape 子图中的 Gather index、Unsqueeze axes、Concat 常量通常来自 initializer，所以执行 Shape 子图前必须已经完成：

```text
内存分配
        ↓
initializer 复制
        ↓
Shape 子图执行
```

## 11. 应该阅读的测试

### 11.1 Operator Shape/dtype 测试

```text
test/operators/test_unary.cc
test/operators/test_gather.cc
test/operators/test_concat.cc
test/operators/test_reshape.cc
```

学习如何验证：

```text
inferShape
inferDataType
Operator 属性
非法输入
```

### 11.2 Native CPU Kernel 测试

```text
test/kernels/nativecpu/test_nativecpu_concat.cc
test/kernels/nativecpu/test_nativecpu_transpose.cc
test/kernels/nativecpu/test_nativecpu_elementwise.cc
```

标准测试流程：

```cpp
创建 Runtime 和 Graph
        ↓
创建输入 Tensor 和 Operator
        ↓
Graph.dataMalloc()
        ↓
写入输入数据
        ↓
Runtime.run(graph)
        ↓
检查输出 Shape、dtype 和数值
```

### 11.3 动态内存测试

```text
test/core/test_graph.cc
pyinfinitensor/tests/test_onnxstub.py
```

重点搜索：

```text
dynamic
reallocation
high_watermark
initializer
set_input
```

只需要先确认现有能力：

- Shape 变大时可以扩容；
- Shape 变小时可以复用高水位容量；
- initializer/weight 不会被动态重规划破坏；
-同一个模型实例可以连续运行。

暂时不必深入 LazyAllocator 每个空闲块的合并算法。

## 12. 必须学习的知识

### 12.1 ONNX 模型结构

必须分清：

```text
ValueInfoProto
TensorProto initializer
NodeProto
graph.input
graph.output
opset_import
attribute 与 input 的版本差异
```

### 12.2 Shape inference 和 value inference

```text
Shape inference
  根据 Tensor 元数据推导输出元数据
  例：Unsqueeze 输入 Shape []，输出 Shape [1]

Value inference / computation
  根据输入 Tensor 数据计算输出 Tensor 数据
  例：Unsqueeze 输入数据 4，输出数据 [4]
```

动态 `Reshape` 的输出 Shape 依赖一次 value computation，不能只靠传统 Shape inference。

### 12.3 图依赖

需要掌握：

- 有向无环图；
-拓扑排序；
- Tensor 的 producer/source；
- Tensor 的 consumer/target；
-从某个 Tensor 反向查找依赖闭包；
-控制依赖和数据依赖。

### 12.4 内存规划

需要理解：

```text
Tensor Shape
    ↓
元素数量
    ↓
getBytes()
    ↓
Blob 大小
    ↓
activation 内存布局
```

以及为什么 Shape 改变后需要重新调用 `dataMalloc()`。

### 12.5 C++ 项目知识

需要掌握：

-虚函数和继承；
-模板函数；
- `optional`；
- `shared_ptr` 风格的项目 `Ref`；
- enum 和 dtype dispatch；
-异常/断言；
- `dynamic_cast` 或项目 `as<T>()`；
- pybind11 的方法绑定。

## 13. 只需要理解、不必深入的内容

以下内容知道用途即可：

- LazyAllocator 的具体空闲块算法；
- Blob owner/view 的完整实现；
- PerfKey 和 Kernel tuning；
- CUDA Graph capture generation；
-其他硬件后端的 Runtime 生命周期。

其他后端 Kernel 可以作为算子语义参考，但基础任务只需要 Native CPU。

## 14. 当前阶段不用阅读的内容

```text
src/nnet
分布式通信算子
CUDA Kernel 编程细节
cuDNN
BANG/KUNLUN/ASCEND Runtime
模型超优化规则
复杂符号 Shape 代数
```

## 15. 推荐阅读顺序

按以下顺序最不容易混乱：

1. 在纸上写出贯穿示例每个 Tensor 的 `getDims()`、dtype 和 data；
2. 阅读 `tensor.h`、`tensor_base.h`、`tensor.cc`；
3. 阅读 `operator.h`、`operator.cc`；
4. 阅读 `kernel.h`、`runtime.cc`；
5. 阅读 `ShapeObj` 和 Native CPU unary Kernel；
6. 阅读 `GatherObj` 和其他后端 Gather Kernel；
7. 阅读 `UnsqueezeObj`、`SqueezeObj` 和 CPU memcpy Kernel；
8. 阅读 `ConcatObj` 和 Native CPU Concat Kernel；
9. 阅读 `CastObj` 和 CUDA Cast Kernel；
10. 阅读 `ReshapeObj` 和 Native CPU Reshape Kernel；
11. 阅读 `GraphObj::topo_sort()`；
12. 阅读 `GraphObj::shape_infer()`；
13. 阅读 `GraphObj::dataMalloc()` 的入口和整体契约；
14. 阅读 `GraphHandler` 和 FFI；
15. 阅读 `onnx.py` 对应算子分支和初始化流程；
16. 阅读 Operator、Kernel、Graph、ONNX 四层测试。

## 16. 建议的阶段性练习

### 练习一：手工区分 Shape 元数据和数据

对下面每个 Tensor 写出 `getDims()`、dtype 和 data：

```text
X = float Tensor，实际 Shape [4, 2, 3]
S = Shape(X)
G = Gather(S, 0)
U = Unsqueeze(G, [0])
R = Concat(U, [6])
Y = Reshape(X, R)
```

正确答案：

| Tensor | `getDims()` | dtype | data |
|---|---|---|---|
| X | `[4,2,3]` | Float32 | 24 个输入值 |
| S | `[3]` | Int64 | `[4,2,3]` |
| G | `[]` | Int64 | `4` |
| U | `[1]` | Int64 | `[4]` |
| R | `[2]` | Int64 | `[4,6]` |
| Y | `[4,6]` | Float32 | 与 X 相同的 24 个元素布局 |

### 练习二：实现独立 Shape CPU Kernel 草稿

不接入项目，先写出伪代码：

```cpp
auto input = op->getInputs(0);
auto output = op->getOutput();
auto dims = input->getDims();
auto out = output->getRawDataPtr<int64_t *>();

for (size_t i = 0; i < dims.size(); ++i)
    out[i] = static_cast<int64_t>(dims[i]);
```

然后回答：

- 为什么不需要读取 `input->getRawDataPtr()`？
-为什么 output dtype 必须是 Int64？
-为什么 output Shape 是 `[input_rank]`？

### 练习三：画出动态 Reshape 的阶段顺序

自己画出并解释：

```text
set_input
    ↓
第一次 dataMalloc
    ↓
runShapeSubgraph
    ↓
refreshResolvedShape
    ↓
shape_infer
    ↓
第二次 dataMalloc
    ↓
runFullGraph
```

## 17. 阅读完成判定

开始实现前，应当能够独立回答以下问题：

### 问题一

为什么 `Shape(X)` 的输出 Tensor Shape 是 `[rank]`、dtype 是 Int64，而输出数据才是 X 的实际维度？

### 问题二

为什么动态 `Reshape` 不能只修改 `ReshapeObj::inferShape()`，而需要增加 Shape 子图预执行阶段？

### 问题三

为什么 Shape 子图执行完成以后，还要再次调用：

```text
shape_infer()
dataMalloc()
```

### 问题四

为什么不能在任意一次 `inferShape()` 中直接读取 Shape Tensor 的 Blob？如何避免读到上一次执行的旧值？

### 问题五

为什么 Shape Tensor 链会暴露 Native CPU `Gather`、`Concat`、`Cast` 的 Int64 支持缺口？

能够完整回答这些问题，就已经具备开始实现基础 Shape Tensor 算子和 Tensor 驱动动态 `Reshape` 所需的核心知识。
