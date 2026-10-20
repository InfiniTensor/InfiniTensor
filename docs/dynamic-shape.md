# ONNX 动态 Shape 子图支持

本实现把小型 Shape Tensor 计算从数据图中提取为主机执行计划，数据计算支持 Native CPU
和 CUDA。同一个 `OnnxStub` 和 C++ Graph 可连续处理不同合法输入 Shape。
除基础任务外，还验证了 Shape 子图部分求值、SqueezeNet 1.0 动态 H/W、CUDA、
CUDA Graph 缓存与失效，以及高水位内存容量复用。

## 构建与复现

以下命令在仓库根目录、已激活的 `infinitensor` Conda 环境中执行。

```bash
conda activate infinitensor
make build TYPE=Debug CUDA=OFF TEST=ON
cp build/Debug/backend*.so pyinfinitensor/src/pyinfinitensor/
python -m pip install -e pyinfinitensor
ctest --test-dir build/Debug --output-on-failure -j4
python -m unittest discover -s pyinfinitensor/tests -p 'test_*.py' -v
python -m pyinfinitensor.dynamic_shape_demo --repeats 1000 --output artifacts/dynamic-shape
python -m pyinfinitensor.real_model_demo --output artifacts/real-model
```

CUDA 使用独立目录，避免覆盖 CPU 构建：

```bash
cmake -S . -B build/CudaDebug -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" -DUSE_CUDA=ON -DBUILD_TEST=ON
cmake --build build/CudaDebug -j8
ctest --test-dir build/CudaDebug --output-on-failure -j4
cp build/CudaDebug/backend*.so pyinfinitensor/src/pyinfinitensor/
python -m pyinfinitensor.cuda_dynamic_shape_demo --output artifacts/cuda
```

验证环境：Python 3.10.21、NumPy 2.2.6、ONNX 1.22.0、ONNX Runtime 1.23.2。
Demo 显式生成 opset 18 / IR 8 模型，避免使用本机 ONNX 默认 IR 13 而导致 ORT 不兼容。
首次准备新环境还需安装 `numpy onnx onnxruntime onnxsim`；已有可用环境无需升级。

Demo 输出 `batch.onnx`、`hw.onnx`、`results.json`，运行任一断言失败都会返回非零退出码。
测试已经接入 `make test-onnx`，C++ 测试追加在已有 `test_reshape` 中，无需新建 CMake 目标。

## 使用接口

```python
import numpy as np
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor.dynamic_shape_demo import make_dynamic_model

stub = OnnxStub(make_dynamic_model(), backend.cpu_runtime())
for batch in [1, 2, 8, 3, 1]:
    x = np.ones((batch, 2, 3), dtype=np.float32)
    stub.set_input([list(x.shape)])
    stub.inputs['x'].copyin_numpy(x)
    stub.run()
    y = stub.outputs['y'].copyout_numpy()
    print(x.shape, y.shape)
print(stub.input_schema)
print(stub.shape_program_stats)
```

`set_input()` 的参数仍按模型输入顺序排列。先设置 Shape，再拷贝本次输入，最后运行。
每轮都使用同一个 stub；请勿只调用低层 `handler.change_shape()`，否则会绕过主机 Shape 计划。
模型若不能以动态维度默认值 1 初始化，可在构造时传入
`input_shapes=[[4, 2, 3]]`；其约束检查与 `set_input()` 相同。
用 `optimize_shape_program=False` 关闭部分求值，可对比原始运行时 Shape 计算量。

## 设计与源文件

| 文件 | 职责 |
| --- | --- |
| `pyinfinitensor/src/pyinfinitensor/shape_program.py` | 维度声明、输入校验、子图提取、部分求值、CPU Shape 求值 |
| `pyinfinitensor/src/pyinfinitensor/onnx.py` | 导入、Shape Tensor 物化、每轮更新、异常恢复、导出 |
| `include/operators/reshape.h` / `src/operators/reshape.cc` | 保留静态接口，增加真实双输入 Reshape |
| `include/core/graph_handler.h` / `src/core/graph_handler.cc` | 双输入算子入口、允许 scalar Shape 更新 |
| `src/ffi/ffi_infinitensor.cc` | 绑定 `reshape_tensor` 和 Tensor 独立存储分配接口 |
| `pyinfinitensor/src/pyinfinitensor/dynamic_shape_demo.py` | 模型构造、ORT 对齐、优化对比、结果保存 |
| `pyinfinitensor/src/pyinfinitensor/real_model_demo.py` | SqueezeNet 1.0 特征网络、五组动态 H/W、ORT 对齐 |
| `pyinfinitensor/src/pyinfinitensor/cuda_dynamic_shape_demo.py` | 普通 CUDA、CUDA Graph 缓存/失效和显存复用验证 |
| `src/operators/pooling.cc` | 每轮刷新 Pooling 的 N/C/H/W，使真实视觉网络可传播动态 H/W |
| `include/core/lazy_allocator.h` 等 | 只读暴露计划字节数、池容量和底层存储 ID |

输入声明使用 `int / str / None` 区分固定维度、符号维度、匿名未知维度。
C++ Tensor 始终存储本次具体整数 Shape。验证 Rank、固定维度、同名符号一致性，拒绝 bool、
非整数、非正数和超过 int32 上限的输入维度。零长度输入尚不在本次支持范围，给出明确错误。

导入时从 Reshape 的第二输入反向追溯 Shape 子图，按 ONNX 拓扑顺序建立求值计划。
`Shape` 从图输入或 initializer 的元数据读维度，不需要读取输入的浮点数据。
支持的主机 Shape 算子为 Shape、Gather、Unsqueeze、Squeeze、Concat、Cast，以及辅助的
Constant 和 Identity。这些操作通过 NumPy 执行；并非为所有后端新增通用数据算子 Kernel。
Gather 支持负索引；Shape 支持 start/end；Cast 支持 int32/int64 目标；axes 必须可静态确定。

动态目标以真实 int64 Tensor 传给 `ReshapeObj` 的第二输入。C++ `inferShape()` 每轮读取
该 Tensor，处理 0 复制维度、单个 -1 推导及元素数量校验；无效值通过异常返回 Python。
静态 initializer 目标继续走原有整数列表接口。`allowzero=1` 当前明确拒绝。

```text
set_input(actual shapes)
  -> validate schemas
  -> evaluate runtime Shape nodes
  -> copy Shape Tensor values
  -> existing Graph::shape_infer()
  -> existing Graph::dataMalloc()
  -> restore initializers and Shape values
copyin_numpy(input data)
run() / run_with_cudagraph() -> existing CPU or CUDA kernels -> outputs
```

Shape Tensor 标记为长生命周期存储以免被激活值生命周期复用覆盖，数值每轮重新写入。
输入数据在内存规划后写入。权重使用既有 `_copy_initializers()` 恢复，继续复用已有
动态池 / naive allocator，而不新建数据图或重新加载模型。
推导失败时恢复旧输入 Shape 和旧 Shape Tensor，重新推导并规划。
调用者在失败后重新写入输入再运行；不承诺保留此前输入或输出内容。

## 部分求值

编译期用 `None` 表示尚未知的维度，已知位置仍保留整数。例如 `[batch, 2, 3]` 表示为
`[None, 2, 3]`；Gather 第 1 项仍可得到常量 2，其 Unsqueeze / Cast / Squeeze 链继续折叠。
运行时只重新计算依赖 batch 的部分，绝不会把初始 batch=1 当成编译期常量。
单独跟踪实际求值 dtype，防止在 object 数组部分求值后把 int32 错误物化成 int64。

Demo 的 10 个 Shape 节点中，固定通道分支的 6 个节点被折叠，运行时求值节点变为 4 个。
统计不包含 2 个数据节点，不表示整个 ONNX 文件删掉 6 个节点；导出保留原始动态图语义。
测时只测 ShapeProgram.evaluate，包含结果小数组复制，不包含数据计算和内存规划。
节点计算量降低是主要证据；微秒级计时随机器负载变化，不用来宣称端到端固定倍数加速。

## 验证与边界

- 同一实例连续 batch 1、2、8、3、1；另有五组同时变化的人工 H/W。
- SqueezeNet 1.0 完整特征提取拓扑使用固定随机权重（非预训练权重），连续运行
  36×36、52×68、68×52、36×52、36×36，并与 ORT 对齐。
- 普通 CUDA 与 ORT 对齐；CUDA Graph 对 4 个不同 Shape 捕获 4 次，第二轮不增加捕获。
  batch 扩大到 32 后底层存储 ID 改变，旧缓存清空并重新捕获，回到 batch=1 也不会误用旧地址。
- 200 轮、共 1000 次内存策略对比记录 exact-fit 与 high-water reuse 的存储更换次数、
  端到端中位延迟和峰值池容量。
- dynamic pool / naive allocator 与优化开关交叉测试；增加、缩小、重复形状均覆盖。
- 与 ORT 比较 Shape、dtype、数值；整数精确相等，浮点 rtol=1e-4、atol=1e-5。
- 独立验证每个 Shape 中间值、权重保持、trim 后再运行、非法输入和异常恢复。
- C++ 验证双输入 Reshape 更新、0/-1、非整除、多个 -1 和真实 CPU 执行。

目前不支持数据相关动态 Shape（如 NonZero）、任意中间数据算子的 Shape 源、动态 Rank、
控制流、用户直接提供内容未知的 Shape 输入，以及动态 Expand / ConstantOfShape。
Shape 输入范围是图输入或 initializer；这些限制在导入阶段报错。
SqueezeNet 使用公开的 1.0 特征网络结构，但权重是确定性随机初始化，不能用于评价分类精度；
选择的 H/W 保证三次 ceil-mode MaxPool 窗口整除，因为现有 MaxPool 在其他边界尺寸上与
ONNX Runtime 存在取整差异。容量复用算法来自项目现有分配器，本次工作是把它接入动态
ONNX 链路、补充状态观测与策略 Benchmark，不把既有算法归为新发明。
已有静态正常路径保留；旧测试中需要改变 batch 的模型改成符号 batch 声明以符合 ONNX 约束。

## 手动提交

先运行上述完整测试，再执行 `git diff --check`，只暂存本项目文件，保留 `cli.md` 未跟踪。
建议一个实现提交，例如 `Support ONNX runtime shape subgraphs`。
PR 标题按要求使用：【训练营】ONNX 动态 Shape 子图编译与执行支持。
报告首页的 PR 链接需在创建 PR 后填写；报告署名为 `uel0p`。

参考：[ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html)、
[Shape](https://onnx.ai/onnx/operators/onnx__Shape.html)、
[Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)、
[Reshape](https://onnx.ai/onnx/operators/onnx__Reshape.html)。
