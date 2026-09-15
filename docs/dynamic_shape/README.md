# ONNX 动态 Shape 子图

本项目在现有 Graph、LazyAllocator 和 CUDA Graph 缓存之上增加 C++ Shape 子图执行阶段。一个 `OnnxStub` 可连续处理不同合法输入 Shape；支持 CPU、CUDA 和 CUDA Graph，数值验证使用独立的 ONNX Runtime CPU Session。

## 构建与测试

以下命令均从 InfiniTensor 仓库根目录运行。先激活带 Python 开发头文件的环境，安装 CMake、C++17 编译器及项目原有构建依赖。若缺少 libdw，可配置 `-DUSE_BACKTRACE=OFF`。

```bash
python -m pip install -r examples/dynamic_shape/requirements.txt
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF -DBUILD_TEST=ON -DBUILD_TEST_ONNX=ON \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/Release -j6
export PYTHONPATH="$PWD/build/Release:$PWD/pyinfinitensor/src"
export OMP_NUM_THREADS=1
ctest --test-dir build/Release --output-on-failure
python -m unittest discover -s pyinfinitensor/tests -p 'test_*.py'
```

`BUILD_TEST_ONNX=ON` 将动态 ONNX 测试注册到 CTest。也可执行 `make test-dynamic-shape TYPE=Release`。不需要把新 backend 复制到 site-packages；`PYTHONPATH` 必须首先指向本次构建目录，避免加载旧版本。

开启 backtrace 时，预期触发异常的负例会打印 C++ 调用栈；请以测试汇总和进程退出状态判断测试结果。

CUDA 构建需要 CUDA Toolkit 和 cuDNN：

```bash
cmake -S . -B build/CUDA -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON -DBUILD_TEST=ON \
  -DCUDNN_ROOT=/path/to/cudnn \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/CUDA -j6
export PYTHONPATH="$PWD/build/CUDA:$PWD/pyinfinitensor/src"
INFINITENSOR_TEST_CUDA=1 python pyinfinitensor/tests/test_dynamic_shape.py
```

cuDNN 位于 Python NVIDIA wheel 中时，`CUDNN_ROOT` 可指向该环境的 `site-packages/nvidia/cudnn`。也可单独提供 `CUDNN_INCLUDE_DIR` 和 `CUDNN_LIBRARY`。原有 CUDA Runtime 默认预留 7 GiB workspace，显存预算应包含它；GPU 测试应串行执行。

## Demo 与模型

```bash
python examples/dynamic_shape/demo.py --output /tmp/cpu_batch.json
python examples/dynamic_shape/demo.py \
  --model examples/dynamic_shape/artifacts/digits_cnn.onnx \
  --output /tmp/cpu_digits.json
python examples/dynamic_shape/demo.py --device cuda --cuda-graph \
  --model examples/dynamic_shape/artifacts/digits_cnn.onnx \
  --output /tmp/cudagraph_digits.json
```

默认模型由 `models.py` 用 ONNX API 构造，连续 Batch 为 `1→2→8→3→1`。模型包含 `Shape→Gather→Unsqueeze→Concat→Reshape→MatMul`，Reshape 的目标来自本轮 Shape Tensor 值。

真实模型是使用 UCI 手写数字数据训练的三层 CNN，包含动态全局平均池化及动态 Reshape。仓库携带 ONNX、真实样本、PyTorch 参考 logits 和训练元信息；推理不需要 PyTorch。H/W Demo 连续使用 `[1,1,8,8]、[2,1,12,10]、[4,1,16,16]、[3,1,10,12]、[1,1,8,8]`。单元测试使用真实图像缩放，对齐 ORT，并在原始分辨率比较 PyTorch logits；Demo 使用固定随机种子的数值输入覆盖更广的数值范围。

重新训练与导出：

```bash
python -m pip install torch==2.9.0 scikit-learn==1.7.2
python examples/dynamic_shape/export_digits.py --epochs 80
```

数据集由 scikit-learn 随包提供，无需额外下载。随机种子、划分、准确率和模型 SHA256 记录在 `artifacts/training.json`。96.11% 是原始 8×8 独立测试集准确率；其他分辨率的结果用于编译器正确性验证，不代表已测得对应分辨率分类准确率。

## Python API

```python
import numpy as np
from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub

stub = OnnxStub(model, backend.cpu_runtime(),
                optimize_shapes=True, memory_reuse=True)
for batch in [1, 2, 8, 3, 1]:
    x = np.ones((batch, 2, 3), dtype=np.float32)
    y = stub.infer({"x": x})["y"]
    print(y.shape, stub.memory_stats())
```

原 API 也可使用：`stub.set_input([x.shape])`，随后 `copyin_numpy(x)`、`stub.run()`、`copyout_numpy()`。CUDA Graph 使用 `stub.infer(feeds, cuda_graph=True)` 或 `run_with_cudagraph()`。输入的名字、rank、dtype、固定维度和跨输入同名符号会被检查。`input_specs` 保存 fixed/symbolic/unknown 声明；`getShape` 返回当前实际 Shape。失败的 Shape 推导会恢复前一组 Shape，之后可以继续提供合法输入。

## 支持范围与错误处理

| 项目 | 当前范围 |
|---|---|
| 动态输入 | 固定 rank；正 int32 范围的实际维度；dim_param 与匿名未知维度 |
| Shape | 全量 Shape，输出 int64；有 start/end 的切片版本尚未支持 |
| Gather | int32/int64 indices、负索引、轴和边界校验；CPU 数据复制与主机 Shape 子图 |
| Unsqueeze / Squeeze | 常量 axes（属性或 initializer）；Squeeze 省略 axes 时每轮重新确定 |
| Concat / Cast | Shape 链路的 int32/int64；CPU Cast 支持 float32/int32/int64 所需转换 |
| Reshape | 静态目标与双输入动态目标；rank-one int64 目标；0、单个 -1、allowzero=1 的核心语义 |
| 数据算子 | 复用现有 CPU/CUDA kernel；新增动态全局平均池化衔接 |
| Shape 子图来源 | initializer/Constant、Shape 的元信息及支持的依赖闭包；单个 Shape 计算输出最多 4096 元素 |
| 非目标范围 | 通用数据依赖形状（例如 NonZero）、控制流、动态 rank 输入、运行时 axes、动态 Expand、ConstantOfShape |

本次验收模型使用 opset 13。已有静态前端支持的其他 opset 不因此扩展为完整覆盖。直接作为模型输入提供目标 Shape Tensor 时，导入需要初始值，目前 `OnnxStub` 不提供该初始值接口；本项目支持由输入实际维度驱动的运行时 Shape 子图，C++ 层可直接传入已赋值的目标 Tensor。

固定维度错误应修改模型声明，而不是继续把固定 `1` 当动态 Batch。`Reshape element count mismatch` 表示目标与输入元素数不一致；多个 `-1`、非法负数、溢出和非 int64 目标会明确报错。CUDA 后端只对受支持的 Shape 依赖使用主机求值，不对任意数据计算做 CPU 回退。当前一个实例的动态准备与执行需要调用者串行使用。

## 性能复现

```bash
python examples/dynamic_shape/benchmark.py --repeats 20 --output /tmp/bench_cpu.json
python examples/dynamic_shape/benchmark.py --device cuda --repeats 20 \
  --output /tmp/bench_cuda.json
python examples/dynamic_shape/benchmark.py --device cuda --cuda-graph \
  --repeats 20 --output /tmp/bench_cudagraph.json
```

Benchmark 比较常量折叠开/关与容量复用开/关四组配置，先预热一个序列，再计时 100 次推理。每次同步完成后计时，并检查 ORT 数值。`prepare_ms` 包含输入 Shape 校验、Shape 子图、图推导和分配；`execute_ms` 为数据图执行；`end_to_end_ms` 还包含输入和输出复制。JSON 提供均值、中位数、P95、真实池分配次数和峰值池容量。

内存统计区分逻辑需求、当前容量、峰值容量以及重新分配时旧池和新池同时存活的字节数；不计 Python/ORT 内存、库内部开销和 CUDA 的固定 workspace。`trim_memory()` 可以显式释放多余池容量，并使受影响的 CUDA Graph 状态失效。缓存复用依赖 Shape、拓扑和实际存储地址均匹配，扩容后历史 Shape 需要在新存储上再次捕获。

## 实现位置与上游基础

`dynamic_shape.py` 保存维度契约并做保守部分求值；`onnx.py` 完成 ONNX 导入和调用衔接；`src/core/shape_executor.cc` 使用真正的 C++ CPU kernel 执行 Shape 依赖；`ReshapeObj` 读取双输入目标；`GraphObj` 在分配后上传已准备的 Shape Tensor，并在生命周期规划中保护它们。

容量复用和 CUDA Graph LRU 缓存来自基线提交 `5cb7c1fc` 已有的上游能力。本项目完成 Shape 子图与这些能力的集成，增加容量复用开关、分配统计和端到端回归；不将已有 allocator 或 CUDA Graph 缓存算法声称为本次新实现。

运行 Demo 和 Benchmark 时，可用 `--output` 将本机测量结果保存到指定位置。提交 PR 时应提供测试摘要、运行环境和性能对比；完整实验日志与项目报告可作为独立附件保存。
