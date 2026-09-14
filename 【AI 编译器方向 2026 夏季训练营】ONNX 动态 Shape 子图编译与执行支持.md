# 【AI 编译器方向 2026 夏季训练营】ONNX 动态 Shape 子图编译与执行支持

项目仓库：https://github\.com/InfiniTensor/InfiniTensor

# 一、项目背景

InfiniTensor 是一个面向 GPU 和 AI 加速器的高性能推理框架，提供计算图、算子、Runtime、内存管理以及多硬件后端等基础能力，并支持从 ONNX 模型导入计算图。

在 AI 编译器中，Tensor Shape 是连接模型前端、算子推导、内存规划和 Runtime 执行的重要信息。对于静态模型，输入和中间 Tensor 的 Shape 在模型加载后即可确定；但在实际推理场景中，Batch Size、图片分辨率以及序列长度等维度经常会随输入发生变化。

ONNX 中除了可以通过 `dim_param` 描述动态维度，还可能通过：

`Shape`、`Gather`、`Unsqueeze`、`Concat`、`Reshape`、`Expand`、`ConstantOfShape`

等算子在计算图内部动态构造目标 Shape。例如：

```Plain Text
Input X
  │
  ├── Shape ── Gather ── Unsqueeze ──┐
  │                                  │
  └─────────────────────────────── Concat ── Reshape
```

这意味着编译器不仅需要能够修改模型输入 Shape，还需要处理**图中的 Shape Tensor 及其依赖关系**：

```Plain Text
动态输入 Shape
      ↓
Shape Tensor 计算
      ↓
确定中间 Tensor Shape
      ↓
更新内存规划
      ↓
执行数据计算
```

InfiniTensor 当前已经具备一定的动态 Shape 基础能力，例如图级 Shape Inference、输入 Shape 修改以及 Shape 改变后的内存重新规划等，但对于 ONNX 模型中由 Shape Tensor 驱动的动态 Shape 计算链路仍有进一步完善空间。

本项目希望在现有能力基础上，补齐 ONNX 动态 Shape 模型从**模型导入、Shape Tensor 计算、Tensor Shape 更新、内存规划到 Runtime 执行**的关键链路，使同一个模型实例能够正确处理不同 Shape 的输入。

# 二、项目目标

完善 InfiniTensor 的 ONNX 动态 Shape 支持，并跑通一个包含动态 Shape 子图的 ONNX 模型。

主要目标包括：

1. 完善 ONNX 动态维度的导入和表示，正确处理固定维度与动态维度；

2. 支持项目所需的基础 Shape Tensor 相关算子；

3. 支持由运行时 Shape Tensor 驱动的 `Reshape` 等动态 Shape 算子；

4. 将 Shape 计算与现有 Shape Inference、Tensor 和内存管理机制正确衔接；

5. 同一个模型实例能够连续处理不同合法输入 Shape，无需重新加载 ONNX 模型；

6. 支持 CPU 后端执行，并完成与 ONNX Runtime 的数值正确性验证；

7. 相关代码、测试和 Demo 通过 PR 提交至 InfiniTensor。

最终期望形成如下执行流程：

```Plain Text
Load ONNX Model
      ↓
Parse Dynamic Shape
      ↓
Input
      ↓
Shape Tensor Computation
      ↓
Update Tensor Shape
      ↓
Memory Planning
      ↓
Data Computation
      ↓
Output
```

# 三、任务拆解

## 1\. 完善 ONNX 动态维度表示

结合 InfiniTensor 当前 ONNX 前端和 Tensor Shape 表示，完善动态维度的导入和运行时处理。

至少需要区分：

- 固定维度，例如 `[1, 3, 224, 224]` 中的 `3`；

- ONNX `dim_param` 描述的符号维度；

- 未指定具体值的未知维度；

- 每次模型执行时输入 Tensor 的实际 Shape。

例如：

```Plain Text
images: ["batch", 3, "height", "width"]
```

模型导入后应能够知道：

```Plain Text
batch / height / width：动态维度
3：固定维度
```

实际执行：

```Plain Text
run 1: [1, 3, 224, 224]
run 2: [4, 3, 224, 224]
run 3: [2, 3, 320, 320]
```

动态维度允许变化，而固定维度发生非法变化时应给出明确错误。

具体实现可以结合 InfiniTensor 现有 Tensor / Graph 结构设计，无需建立复杂的符号表达式系统，重点保证本项目所需动态 Shape 信息能够正确保存和使用。

## 2\. 补齐 Shape Tensor 相关算子

结合项目动态 Shape 模型实际使用的计算链路，补齐必要的 ONNX Shape 相关算子。

基础范围建议至少支持：

- `Shape`

- `Gather`

- `Unsqueeze / Squeeze`

- `Concat`

- `Cast`

- 动态 `Reshape`

根据实际模型需要，可以进一步支持：

- `ConstantOfShape`

- `Expand`

重点不是单独增加若干算子，而是至少跑通一条完整的 Shape Tensor 计算链路，例如：

```Plain Text
Input
  ↓
Shape
  ↓
Gather
  ↓
Unsqueeze
  ↓
Concat
  ↓
Reshape
```

其中 `Reshape` 的目标 Shape 必须能够来自**运行时计算得到的 Tensor**，而不是要求在 ONNX 模型导入阶段就能够静态确定。

新增或完善的算子需要提供必要的单元测试，并重点验证：

- Shape 推导；

- dtype；

- 数值结果；

- 关键异常输入。

实现范围以满足项目动态 Shape 模型为主，无需覆盖相关 ONNX 算子的全部 opset 和所有参数组合。

## 3\. 打通动态 Shape Runtime 执行流程

将 Shape Tensor 计算与 InfiniTensor 现有 Graph Shape Inference、Tensor 和内存管理机制进行集成。

一次动态模型执行应能够完成：

```Plain Text
输入 Tensor
    ↓
获取本次实际 Shape
    ↓
执行必要的 Shape Tensor 计算
    ↓
确定中间 Tensor / 输出 Tensor Shape
    ↓
更新或重新规划激活值内存
    ↓
执行数据计算
    ↓
返回结果
```

需要重点保证：

- Shape 相关计算先于依赖其结果的数据计算执行；

- Tensor Shape 改变后，对应内存能够正确更新；

- 输入变大时不会继续使用容量不足的旧内存；

- 输入变小时不会读取上一次执行的残留数据；

- initializer / 权重不会因为动态内存重新规划而被破坏；

- 静态 ONNX 模型原有执行路径保持兼容。

优先复用 InfiniTensor 已有的 Shape Inference 和动态内存管理能力，避免重新实现已有机制。

## 4\. 完成动态 ONNX 模型端到端验证

准备至少一个包含运行时 Shape 计算的 ONNX 模型，完成端到端动态 Shape 推理。

模型可以是：

- 人工构造的典型动态 Shape ONNX 模型；

- 动态 Batch 的轻量模型；

- 动态图片 H/W 的视觉模型；

- 动态序列长度的轻量 NLP 模型。

基础任务不要求模型规模较大，重点是能够覆盖：

```Plain Text
动态输入
    ↓
Shape 子图
    ↓
动态 Reshape / 其他数据算子
    ↓
动态输出
```

同一个模型实例至少连续运行五组不同输入 Shape，例如：

```Plain Text
bs = 1
bs = 2
bs = 8
bs = 3
bs = 1
```

运行过程中不得重新加载 ONNX 模型。

## 5\. 完成 ONNX Runtime 数值正确性验证

使用相同 ONNX 模型和输入，与 ONNX Runtime 进行结果对齐。

至少比较：

- 输出 Tensor Shape；

- 输出 Tensor 数值。

浮点结果可以参考：

```Plain Text
np.testing.assert_allclose(
    infinitensor_output,
    ort_output,
    rtol=1e-4,
    atol=1e-5,
)
```

不同设备、dtype 和 Kernel 实现可能引入不同浮点误差，可以根据实际情况选择合理阈值，但需要在项目报告中说明误差标准和判定依据。

同时需要覆盖动态 Shape 的连续变化，例如：

```Plain Text
1 → 2 → 8 → 3 → 1
```

验证 Shape 更新、内存规划和计算结果在连续执行过程中始终正确。

# 四、提交要求与评判标准

## 提交要求

**项目提交截止时间**：2026/09/20\.

请提交以下内容：

1. **项目代码**

    1. 项目完成后，将代码通过 PR 提交至 InfiniTensor，PR 命名：`【训练营】ONNX 动态 Shape 子图编译与执行支持`。

    2. 提交 PR 前，请将项目相关代码整理为一个或少量逻辑清晰的 Commit，便于项目评审。

    3. PR 中应包含：

        - ONNX 前端相关修改；

        - Graph / Operator / Runtime 等必要实现；

        - 新增或完善的算子实现；

        - 必要且有代表性的单元测试；

        - 动态 Shape 子图测试；

        - 端到端动态模型 Demo；

        - CMake / 测试系统等必要集成。

    4. 测试应重点覆盖核心功能和关键边界场景，避免针对等价参数组合、重复执行路径或已经被其他测试覆盖的行为增加大量冗余测试。

2. **项目报告**

    1. **项目报告提交要求**

        - 报告以 **PDF 格式**提交； 

        - 通过邮件发送至 `zhushuang@qiyuanlab.com`，以附件形式提交； 

        - 报告文件命名参考：`【2026夏季InfiniTensor训练营- AI编译器方向】姓名_ONNX动态Shape项目报告.pdf`； 

        - 邮件主题命名参考：`【2026夏季InfiniTensor训练营- AI编译器方向】姓名_ONNX动态Shape项目报告`；

        - 项目报告首页需注明以下信息：

            - 项目名称：ONNX 动态 Shape 子图编译与执行支持 

            - 项目提交人：XXX 

            - GitHub PR 对应地址：[`https://github.com/InfiniTensor/InfiniTensor/pull/xxx`](https://github.com/InfiniTensor/InfiniTensor/pull/xxx)

    2. 项目报告至少包含：

        1. **设计与实现，**介绍动态 Shape 支持的整体方案，包括

            - ONNX 动态维度表示；

            - Shape Tensor 及相关算子实现；

            - 动态 Shape 执行流程；

            - Tensor Shape 更新；

            - 动态内存处理；

            - 与现有静态执行流程的兼容方式。

        2. **结果展示**

            - 算子测试结果；

            - Shape 子图测试结果；

            - 连续不同输入 Shape 的执行结果；

            - InfiniTensor 与 ONNX Runtime 的输出 Shape 和数值对比。

        3. **使用说明**

            - ONNX 模型准备方式；

            - Demo 启动方式；

            - 动态输入设置方式；

            - 当前支持范围；

            - 已知限制及常见错误处理。

## 通过标准

满足以下要求视为通过：

- 能够正确导入并表示项目所需的 ONNX 动态维度；

- 至少支持一条完整的 Shape Tensor 计算链路：

```Plain Text
Shape → Gather → Unsqueeze → Concat → Reshape
```

- `Reshape` 的目标 Shape 可以来自运行时 Tensor；

- 同一个模型实例能够连续处理至少五组不同输入 Shape；

- Shape 更新后能够正确完成内存规划和数据执行；

- 输入 Shape 由小变大、由大变小时均能正确运行；

- 与 ONNX Runtime 的输出 Shape 和数值结果满足报告中约定的误差标准；

- 新增或修改功能具有必要且有代表性的自动化测试；

- 现有相关静态模型测试保持通过；

- Demo 和测试可以按照文档说明复现。

## 优秀标准

在完成上述要求的基础上，可进一步完成以下一个或多个方向。

### 1\. Shape 子图编译期优化

识别模型中的 Shape 计算子图，对其中能够在编译期确定的节点进行常量折叠或部分求值。

例如：

```Plain Text
Shape Subgraph
      ↓
Compile-time Known ──→ Constant Folding
      ↓
Runtime Dependent ───→ Runtime Shape Computation
```

并能够展示优化前后的 Shape 子图节点数量或运行时 Shape 计算量。

### 2\. 真实动态 H/W 或动态序列模型

跑通真实模型中的动态图片 H/W 或动态序列长度，而不仅是 Batch 维变化。

要求至少连续执行五组不同 Shape，并与 ONNX Runtime 完成结果验证。

### 3\. CUDA 动态 Shape 支持

在 CPU 基础能力完成后，进一步支持 CUDA 后端动态 Shape 执行，验证：

- Shape 变化后的显存规划正确；

- 不发生非法内存访问；

- 连续 Shape 变化结果正确；

- 与 CPU / ONNX Runtime 结果一致。

### 4\. CUDA Graph 与动态 Shape 集成

结合 InfiniTensor 已有 CUDA Graph 机制，使不同 Shape 对应的执行状态能够正确捕获和复用。

需要保证：

- Shape 或内存地址变化后不会错误复用旧 CUDA Graph；

- 新 Shape 可以正确重新捕获；

- 已经出现过的 Shape 可以复用对应执行状态；

- CUDA Graph 和普通 CUDA 执行结果一致。

### 5\. 动态内存容量复用与性能优化

区分 Tensor 的逻辑 Shape 和实际分配容量。

例如已经为：

```Plain Text
[8, 1024]
```

分配过足够大的内存后，再执行：

```Plain Text
[4, 1024][2, 1024]
```

时可以尝试直接复用已有存储，减少动态 Shape 场景下频繁的内存申请与释放。

通过 Benchmark 对比优化前后的：

- 内存重新分配次数；

- 动态 Shape 执行延迟；

- 峰值内存 / 显存；

- 端到端推理性能。

最终能够形成：

```Plain Text
ONNX Dynamic Shape
        ↓
Shape Tensor Computation
        ↓
Compile-time / Runtime Shape Processing
        ↓
Dynamic Memory Planning
        ↓
CPU / CUDA Execution
        ↓
ONNX Runtime Validation
```

使动态 Shape 能力真正贯穿 InfiniTensor 的**ONNX 前端、计算图、Shape 推导、内存规划和 Runtime 执行**。

