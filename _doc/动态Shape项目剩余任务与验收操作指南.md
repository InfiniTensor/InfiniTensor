# 动态 Shape 项目：剩余任务与验收操作指南

## 1. 先明确现在该做什么

本指南按当前提交 `2b4e0fb`、`64df953` 的代码整理。你已经写了动态维度表示、基础 Shape Tensor 算子，以及动态 Reshape 的前端、算子、Runtime 和内存衔接代码。

**接下来不是把七个目标重新实现一遍，而是：补固定维度约束 → 用测试验证动态链路 → 修复实际暴露的问题 → 整理 Demo、PR 和报告。**

代码存在不等于验收完成。特别是原来的静态 `test_reshape` 通过，不能证明运行时 Shape Tensor 路径正确。

本文只给出操作方案和待加入的代码，没有替你修改源码；下文新测试和 Demo 需要你落到指定文件后运行，不能把示例输出当成已通过记录。

| 目标 | 当前应如何看待 | 还需要的证据 |
| --- | --- | --- |
| 固定、符号、未知维度表示 | 已有表示与导入代码；固定维度约束还需补齐 | 合法变化成功，固定维度变化明确报错 |
| 基础 Shape Tensor 算子 | 已有主要实现 | Shape、dtype、数值和关键错误测试 |
| 运行时 Shape Tensor 驱动 Reshape | 已接入主要路径 | 改变 Shape Tensor 的值，输出 Shape 随之变化 |
| Shape Inference、Tensor、内存衔接 | 已有实现，仍需动态验证 | 变大、变小、初始化常量、后续计算均正确 |
| 同实例多 Shape | 需要端到端证明 | 一个 OnnxStub 连续运行 `1 → 2 → 8 → 3 → 1` |
| CPU 与 ONNX Runtime 对齐 | 需要新增对比 | 每轮实际输出 Shape、数值均通过 |
| PR、测试、Demo | 已有代码提交，不等于已完成提交要求 | 可复现命令、测试、Demo、PR 和 PDF 报告 |

暂时不要扩展 CUDA、CUDA Graph、真实大模型、符号表达式引擎、性能优化。`Expand`、`ConstantOfShape` 按模型需要扩展，不是这条最小链路的前置任务。

## 2. 准备环境：统一在 WSL 中运行

下面所有 shell 命令都是 **WSL Bash 命令**，不要直接粘贴进 Windows PowerShell。Windows 文件目录对应 WSL 的 `/mnt/d/...`。

```bash
cd /mnt/d/InfiniTensor/Advance/InfiniTensor
export PY=/home/amantiras/miniconda3/bin/python3
export PYTHONPATH="$PWD/build/Debug:$PWD/pyinfinitensor/src${PYTHONPATH:+:$PYTHONPATH}"

git status --short
"$PY" -c 'import sys, numpy, onnx; print(sys.executable); print(numpy.__version__); print(onnx.__version__)'
"$PY" -c 'import backend; print(backend.__file__)'
"$PY" -c 'import onnxruntime as ort; print(ort.__version__)'
```

这里特意使用当前构建对应的 Conda Python，而不是系统 `python3`。`backend.__file__` 应指向 `build/Debug/backend...so`，避免测试到旧的扩展库。重新打开终端后，需要重新设置上面两个环境变量。

如果只有 `onnxruntime` 缺失，再安装它：

```bash
"$PY" -m pip install onnxruntime
```

使用已经配置好的 `build/Debug`，不要改用仓库根目录的 `build` 缓存，也不需要删除构建目录。

## 3. 第一件事：补上固定维度不能变化的要求

### 3.1 为什么这里仍然需要检查

任务书明确要求：固定维度发生非法变化时，给出明确错误。这不是额外设计一套复杂校验框架。

你可以不增加 `_validate_input_shapes` 函数，直接在 `set_input` 里完成最小检查。关键是**先检查所有输入，再修改 Tensor**，避免第二个输入非法时第一个输入已经被改掉。

修改 `pyinfinitensor/src/pyinfinitensor/onnx.py` 的 `OnnxStub.set_input`。当前文件已有 `operator` 导入；保留它：

```python
def set_input(self, inputShapes: List[Sequence[int]]) -> None:
    if len(inputShapes) != len(self.inputs):
        raise ValueError(
            "inputShapes must contain one shape per model input; expected "
            "{}, got {}".format(len(self.inputs), len(inputShapes))
        )

    normalized_shapes = []
    for name, supplied_shape in zip(self.inputs, inputShapes):
        spec = self.input_shape_specs[name]
        if len(supplied_shape) != len(spec):
            raise ValueError(
                f"Input {name}: expected rank {len(spec)}, "
                f"got {len(supplied_shape)}"
            )

        shape = []
        for axis, (value, dimension) in enumerate(zip(supplied_shape, spec)):
            try:
                actual = operator.index(value)
            except TypeError as exc:
                raise ValueError(
                    f"Input {name}, axis {axis}: dimension must be an integer"
                ) from exc
            if actual < 0:
                raise ValueError(
                    f"Input {name}, axis {axis}: dimension must be non-negative"
                )
            if dimension.kind == DimensionKind.FIXED and actual != dimension.value:
                raise ValueError(
                    f"Input {name}, axis {axis}: fixed dimension "
                    f"must be {dimension.value}, got {actual}"
                )
            shape.append(actual)
        normalized_shapes.append(shape)

    for name, shape in zip(self.inputs, normalized_shapes):
        self.handler.change_shape(shape, self.inputs[name].fuid())
    self.handler.shape_infer()
    self.init()
    self.current_input_shapes = {
        name: tuple(shape)
        for name, shape in zip(self.inputs, normalized_shapes)
    }
```

`operator.index` 接受整数和 NumPy 整数，不会把 `2.5` 静默截断成 `2`。动态维度接受本次具体非负整数；固定维度包括固定的 `0`，不能把 `0` 当作“未知”。这段不实现跨输入的符号等值约束，也不保证任意后续算子失败时整个图自动回滚。

### 3.2 两处旧测试的模型声明必须同步修正

文件 `pyinfinitensor/tests/test_onnxstub.py` 中：

- `test_initializer_is_restored_after_reallocation`
- `test_dynamic_reallocation_restores_initializer`

它们构造了输入、输出均为 `[1, 2]` 的 MatMul，却在后面修改 batch。把**这两个测试内**的输入和输出声明都改为：

```python
[value_info("x", ["batch", 2])],
[value_info("y", ["batch", 2])],
```

不要全文件替换 `[1, 2]`，其他静态测试需要保留固定维度。这里是纠正测试模型的声明，不是削弱测试；权重、执行循环、数值断言都保留。

### 3.3 添加一个小测试

在同文件已有的测试类内添加下列方法，复用现有 `make_model`、`value_info`、`import_model`，放在文件末尾 `unittest.main()` 之前：

```python
def test_fixed_dimension_is_rejected_before_mutation(self):
    model = make_model(
        [helper.make_node("Identity", ["x"], ["y"])],
        [value_info("x", ["batch", 2])],
        [value_info("y", ["batch", 2])],
    )
    stub = import_model(model)
    stub.set_input([[3, 2]])
    before = tuple(stub.getShape("x"))

    with self.assertRaisesRegex(ValueError, "fixed dimension"):
        stub.set_input([[3, 4]])
    self.assertEqual(tuple(stub.getShape("x")), before)

    with self.assertRaisesRegex(ValueError, "rank"):
        stub.set_input([[6]])
    self.assertEqual(tuple(stub.getShape("x")), before)
```

执行：

```bash
"$PY" pyinfinitensor/tests/test_onnxstub.py -v
```

通过标准：合法 batch 可变，固定维度和 rank 错误明确被拒绝，原有 CPU 测试仍通过。仅 CPU 构建下 CUDA 测试跳过正常，不代表 CUDA 已验证。

还应在现有维度表示测试中确认：`["batch", 2, None]` 分别解析为 SYMBOLIC、FIXED、UNKNOWN，固定 `0` 仍为 FIXED；检查声明保存和实际 Shape 更新是两份信息，不互相覆盖。

## 4. 第二件事：先隔离验证动态 Reshape 本身

修改 `test/operators/test_reshape.cc`，在 `namespace infini` 内追加测试。这样不用新建 CMake target，也不用重新配置 CMake：

```cpp
TEST(Reshape, RuntimeShapeTensorChanges) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    for (bool naive : {false, true}) {
        Graph g = make_ref<GraphObj>(runtime);
        auto data = g->addTensor({2, 3}, DataType::Float32);
        auto target = g->addTensor({2}, DataType::Int64);
        data->setInput();
        target->setInput();
        auto op = g->addOp<ReshapeObj>(data, target, nullptr, false);
        op->getOutput()->setOutput();
        g->dataMalloc(naive);

        const vector<float> values{0, 1, 2, 3, 4, 5};
        data->copyin(values);
        for (const auto &spec :
             vector<vector<int64_t>>{{3, 2}, {1, 6}, {0, -1}}) {
            target->copyin(spec);
            runtime->run(g);
            const Shape expected = spec[0] == 0
                                       ? Shape{2, 3}
                                       : Shape{static_cast<int>(spec[0]),
                                               static_cast<int>(spec[1])};
            EXPECT_EQ(op->getOutput()->getDims(), expected);
            EXPECT_TRUE(op->getOutput()->equalData(values));
        }
    }
}
```

这个测试刻意保持数据输入为 `[2, 3]`，只改 `target` 的数据。它证明输出 Shape 不是在导入时写死的，并验证重分配之后输入数据仍然有效。`{0, -1}` 在 `allowZero=false` 时解析为 `{2, 3}`。

```bash
cmake --build build/Debug --target test_reshape backend -j2
ctest --test-dir build/Debug -R '^test_reshape$' --output-on-failure
```

再补少量错误用例，按仓库现有异常测试风格使用 `EXPECT_ANY_THROW`，每个错误场景单独构建图：

| 场景 | 预期 |
| --- | --- |
| Int64 target 数据为 `[-1, -1]` | 拒绝多个待推导维度 |
| 6 个元素，target 为 `[4, 2]` | 拒绝元素数不一致 |
| target dtype 为 Float32 | 构造时拒绝 |
| target Shape 为 `[1, 2]`，即二维 Tensor | 构造时拒绝，不能把“包含两个值”和“一维”混淆 |

只有这个小测试通过后，才进入下面的 ONNX 子图测试。否则先看 `resolveRuntimeShape()`、`prepareRuntimeShapes()` 和重分配逻辑，不必先排查 ONNX 导入。

## 5. 第三件事：建立可以提交的端到端 Demo

### 5.1 模型结构和验收意义

新建 `examples/onnx_dynamic_shape_demo.py`，实现下面的完整代码。模型使用固定 rank、动态 batch：

```text
X[batch, 2, 3] → Shape → Gather → Unsqueeze ─┐
int32[6] → Cast ────────────────────────────┴→ Concat → target[2]

X 的数据 ────────→ Reshape(data=X, shape=target) → Add(bias) → Y[batch, 6]
```

Concat 输出作为 Reshape 第二输入，Add 的 bias 是独立浮点 initializer。`target[2]` 表示有两个元素，其值为 `[batch, 6]`。

Shape 子图不仅出现了，而且其运行时结果真正参与了 Reshape。Add 用于检查后续普通数据算子以及浮点 initializer 在重分配后仍正确。此例不覆盖 Squeeze，它需要独立小测试。

### 5.2 完整 Demo 代码

```python
import argparse

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from pyinfinitensor import backend
from pyinfinitensor.onnx import OnnxStub


def build_model():
    def const(name, value, dtype):
        return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)

    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Gather", ["x_shape", "index"], ["batch_scalar"], axis=0),
        helper.make_node("Unsqueeze", ["batch_scalar", "axes"], ["batch_vector"]),
        helper.make_node("Cast", ["six_i32"], ["six_i64"], to=TensorProto.INT64),
        helper.make_node("Concat", ["batch_vector", "six_i64"], ["target"], axis=0),
        helper.make_node("Reshape", ["x", "target"], ["flat"]),
        helper.make_node("Add", ["flat", "bias"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "runtime_shape_demo",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 6])],
        initializer=[
            const("index", 0, np.int64),
            const("axes", [0], np.int64),
            const("six_i32", [6], np.int32),
            const("bias", [0.25, -1.0, 2.0, 0.5, -0.5, 3.0], np.float32),
        ],
        value_info=[
            helper.make_tensor_value_info("flat", TensorProto.FLOAT, ["batch", 6])
        ],
    )
    # 本人工模型只使用 IR 8 / opset 18 能表达的功能。
    # 不要据此给真实模型随意降 IR 版本。
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8
    )
    onnx.checker.check_model(model)
    return model


def validate_model(naive=False, batches=(1, 2, 8, 3, 1), verbose=True):
    model = build_model()
    # 两个实例都只创建一次，必须在循环外。
    stub = OnnxStub(model, backend.cpu_runtime(), use_naive_allocator=naive)
    reference = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(2026)
    records = []
    for batch in batches:
        x = rng.standard_normal((batch, 2, 3)).astype(np.float32)
        # 顺序不能颠倒：set_input 可能重新分配内存，之后再写输入。
        stub.set_input([list(x.shape)])
        stub.inputs["x"].copyin_numpy(x)
        stub.run()

        expected = reference.run(["y"], {"x": x})[0]
        actual_shape = tuple(stub.getShape("y"))
        if actual_shape != expected.shape:
            raise AssertionError(
                f"Shape mismatch: InfiniTensor={actual_shape}, ORT={expected.shape}"
            )
        # 先比较真实 Shape，再 reshape 数据；不能用期望 Shape 掩盖 metadata 错误。
        actual = np.asarray(
            stub.outputs["y"].copyout_float(), dtype=np.float32
        ).reshape(actual_shape)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        max_abs_error = float(np.max(np.abs(actual - expected)))
        records.append((batch, actual_shape, max_abs_error))
        if verbose:
            print(
                f"batch={batch}, shape={actual_shape}, "
                f"max_abs_error={max_abs_error:.8g}, PASS"
            )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive", action="store_true")
    parser.add_argument("--save-model", default=None)
    args = parser.parse_args()
    if args.save_model:
        onnx.save(build_model(), args.save_model)
    validate_model(naive=args.naive)
```

运行默认内存池，再运行 naive allocator：

```bash
"$PY" examples/onnx_dynamic_shape_demo.py
"$PY" examples/onnx_dynamic_shape_demo.py --naive
```

每次命令应输出五行 PASS，batch 顺序为 `1, 2, 8, 3, 1`，Shape 分别为 `(1,6)`、`(2,6)`、`(8,6)`、`(3,6)`、`(1,6)`。这只是预期，不是已取得的结果。不要在失败时放宽误差阈值或重新创建 OnnxStub 来绕过问题。

需要保存模型供展示时，可以使用 `--save-model build/Debug/dynamic_shape_demo.onnx`。模型可以由脚本生成，不必把二进制 ONNX 文件提交到 Git。

### 5.3 把同一条链路纳入自动化测试

新建 `pyinfinitensor/tests/test_dynamic_shape.py`：

```python
import sys
import unittest
from pathlib import Path

# 定位本仓库的 Demo，不依赖启动命令所在目录。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from onnx_dynamic_shape_demo import validate_model


class TestDynamicShape(unittest.TestCase):
    def test_same_instance_matches_ort(self):
        for naive in (False, True):
            with self.subTest(naive=naive):
                records = validate_model(naive=naive, verbose=False)
                self.assertEqual([record[0] for record in records], [1, 2, 8, 3, 1])


if __name__ == "__main__":
    unittest.main()
```

```bash
"$PY" pyinfinitensor/tests/test_dynamic_shape.py -v
```

Demo 和测试共用验证逻辑，避免复制两套模型构造代码。ORT 是这项验收测试的必要依赖；没有安装时应修复环境，不要静默跳过正确性对比。

## 6. 第四件事：补代表性算子边界测试

不要为所有参数组合写庞大的测试矩阵。先检查现有测试覆盖，再补下面缺口：

| 功能 | 必须检查的内容 | 建议位置 |
| --- | --- | --- |
| Shape | 输入 `[2,3,4]`，输出 Shape `[3]`、dtype Int64、数据 `[2,3,4]` | 新 CPU Shape 测试，或 Python 小模型测试 |
| Gather | Int64 数据和索引；`-1` 取最后元素；越界报错 | 算子推导在 `test/operators/test_gather.cc`，数据执行补 CPU/Python 测试 |
| Unsqueeze/Squeeze | Int64 Shape Tensor 插入/去除大小为 1 的维度；值不变；重复或非法 axes | `test/operators/test_reshape.cc` 与 CPU/Python 小模型 |
| Concat | Int64 `[batch]` 和 `[6]` 得到 `[batch,6]` | Demo 已覆盖主链；需要时在 `test/kernels/nativecpu/test_nativecpu_concat.cc` 补直接值断言 |
| Cast | Int32 `[6]` 变 Int64 `[6]`，dtype 和数据均正确 | Demo 已覆盖该转换；直接检查 dtype 的小测试 |

特别注意：`test_gather` 中的 Shape Inference 测试，不等于 CPU Gather Kernel 的数据正确性测试。Squeeze 不在上面的 Demo 内，必须独立覆盖。

先保证一个完整闭环成功，再补上述边界。若新建 C++ 测试文件，CMake 通过文件 glob 收集，需要先重新配置；只修改已有测试文件则不需要：

```bash
# 仅新增 C++ 测试文件时执行，沿用当前缓存中的构建选项。
cmake -S . -B build/Debug
```

新 target 通常对应文件名去掉 `.cc`；用 `ctest --test-dir build/Debug -N` 确认它确实注册，不能只写了文件却没有运行它。

## 7. 出错时按现象定位，不要同时大改所有层

| 现象 | 优先检查 |
| --- | --- |
| 找不到 `reshape_dynamic` | 是否重建 backend；`backend.__file__` 是否是本次构建 |
| 导入 Reshape 就失败 | `onnx.py` 是否把运行时第二输入交给 `reshape_dynamic`，而不是当 initializer 解析 |
| C++ 小测试失败，ONNX 还没参与 | `src/operators/reshape.cc` 的解析、输出更新；`src/core/runtime.cc` 的执行与内存顺序 |
| 输出 Shape 仍是占位 Shape | Shape producer 是否被收集/执行；`resolveRuntimeShape()` 是否执行；结果是否传播 |
| 第一次成功，第二次或变小时失败 | 旧 Shape 缓存、重分配后的数据保存、输出容量和逻辑 Shape |
| naive 成功、默认内存池失败 | 内存池生命周期、alias/复用、重分配时输入与 initializer 的保留 |
| Shape 正确，但 Add 后数值错 | Reshape 数据搬运、bias 恢复、后续算子 Shape 推导 |
| Gather/Concat/Cast kernel 找不到 | CPU 注册项及支持 dtype，不要只检查算子类有没有定义 |

你当前方案中的占位 Shape、Shape 子图白名单、固定输出 rank 都是需要明确说明的范围限制。通过这个合成模型，不代表任意数据依赖 Shape、多级动态 Reshape 或任意模型都已支持。若测试发现占位 Shape 导致下游构造失败，应修复或明确限制，不能把所有模型宣称为支持。

## 8. 最终回归：按顺序运行这一组命令

源码和测试补好后，先重编译，再跑相关 C++、Python 回归和 Demo。以下 target 来自当前仓库文件名：

```bash
cmake --build build/Debug --target backend test_reshape test_gather test_unary test_concat test_nativecpu_concat test_graph -j2

ctest --test-dir build/Debug \
  -R '^(test_reshape|test_gather|test_unary|test_concat|test_nativecpu_concat|test_graph)$' \
  --output-on-failure

"$PY" pyinfinitensor/tests/test_onnxstub.py -v
"$PY" pyinfinitensor/tests/test_dynamic_shape.py -v
"$PY" examples/onnx_dynamic_shape_demo.py
"$PY" examples/onnx_dynamic_shape_demo.py --naive

git diff --check
git status --short
```

如果你新增了其他 C++ 测试 target，必须加入构建和运行清单。测试失败时先修复对应问题；如果仅有环境依赖缺失，记录原因，不要记成通过。

上面没有要求全仓库测试全跑一遍：部分测试依赖其他设备、模型或配置。先完成与你修改功能有关的回归；PR 的 CI 若暴露其他回归，再处理实际失败。特别是其他历史测试如果也修改固定输入维度，要判断其模型声明是否本来就应该是动态，不能一律绕过固定维度检查。

## 9. 整理 PR 和报告

### 9.1 增加一个简短复现说明

建议新建 `_doc/动态Shape使用与验证.md`，只记录最终已验证的内容：

1. 支持范围：Native CPU、固定 rank、已验证 Shape 子图和 dtype。
2. 环境：实际 Python、ONNX、ORT、编译配置。
3. 构建、测试、Demo 启动命令。
4. 五次运行的真实输出，以及误差标准 `rtol=1e-4, atol=1e-5`。
5. 限制：例如当前 ONNX `allowzero=1` 拒绝、输出 rank 不能动态变化、CUDA 不支持此路径。以最终代码为准。

这里是给评审运行用的说明，不需要再写成长篇学习笔记。删除实现里已经废弃的大段注释代码，保留解释设计原因的注释；不要夹带无关重构。

### 9.2 提交前检查

```bash
git diff --stat
git diff --check
git status --short
git remote -v
git branch --show-current
```

你已有两个功能提交，不必为了“必须一个 commit”冒险重写历史。任务要求是一个或少量逻辑清晰的提交；再增加一个测试、Demo 和必要修正提交是合理路径。

可在当前成果上创建提交分支，例如：

```bash
git switch -c onnx-dynamic-shape-camp
```

逐个暂存你实际修改的文件，例如：

```bash
git add pyinfinitensor/src/pyinfinitensor/onnx.py
git add pyinfinitensor/tests/test_onnxstub.py
git add pyinfinitensor/tests/test_dynamic_shape.py
git add test/operators/test_reshape.cc
git add examples/onnx_dynamic_shape_demo.py
git add _doc/动态Shape使用与验证.md
# 其他新算子测试，按你真正新增的路径逐个 git add。
git diff --cached --stat
git diff --cached --check
git commit -m "Add dynamic shape validation, tests and CPU demo"
```

不要把 `build/`、`.so`、本地环境目录和临时日志一起提交。本指南是学习操作文档，是否放进 PR 可自行决定；评审最需要的是精简复现说明。

确认 `origin` 是自己的 fork 后再推送：

```bash
git push -u origin onnx-dynamic-shape-camp
```

在 GitHub 向 `InfiniTensor/InfiniTensor` 发起 PR。基础分支选择项目实际接收贡献的分支，提交前检查完整 diff，确保没有无关提交。PR 标题按任务书填写：

```text
【训练营】ONNX 动态 Shape 子图编译与执行支持
```

PR 正文写清：实现范围、执行顺序、测试命令与真实结果、Demo 用法、已知限制。不要只写“支持动态 Shape”，也不要把未验证 CUDA 写成支持。

### 9.3 PDF 报告不要遗漏

仓库任务书标注截止时间为 **2026/09/20**。以组织方后续通知为准；当前日期已接近截止，优先保证基本验收闭环。

报告至少包含：

- 首页：姓名、项目名称、PR 地址。
- 设计：动态维度表示，Shape 子图执行，Reshape 解析，Shape 更新与内存重分配，静态路径兼容。
- 结果：算子测试、五次连续输入 Shape、ORT Shape/数值对齐、误差标准。
- 使用：环境和命令、Demo 模型、支持范围与限制。

根据任务书，导出 PDF 后提交到 `zhushuang@qiyuanlab.com`，文件名参考：

```text
【2026夏季InfiniTensor训练营- AI编译器方向】姓名_ONNX动态Shape项目报告.pdf
```

邮件主题参考同名去掉 `.pdf`。先取得 PR 地址，再补全报告首页。这里仅记录任务书要求，不表示 PR 或邮件已经提交。

## 10. 最后用这张清单判断是否完成

- [ ] 固定维度非法变化明确报错；动态维度允许变化。
- [ ] 符号、未知、固定维度信息没有被实际输入 Shape 覆盖。
- [ ] C++ 测试证明同一个 Reshape 使用不同运行时 target 值得到不同输出 Shape。
- [ ] 必要 Shape 算子的推导、dtype、数据和关键错误有代表性测试。
- [ ] 同一个 OnnxStub 连续完成 `1 → 2 → 8 → 3 → 1`，循环内没有重建模型。
- [ ] 每次都先比较真实输出 Shape，再与 ORT 比较数值。
- [ ] 默认内存池和 naive 路径通过，initializer 与下游数据计算正确。
- [ ] 相关静态回归通过；新增测试实际被运行。
- [ ] Demo、复现说明和测试进入 PR。
- [ ] PDF 报告包含真实结果和 PR 地址，按要求提交。

建议工作顺序：先完成第 3、4 节的小闭环，再攻第 5 节完整链路；跑通后补第 6 节代表性边界，最后做回归和提交。不要同时启动新算子、CUDA 和优化三个方向。如果端到端失败，把第一个失败的命令、完整报错、batch 和 allocator 模式记录下来，就能继续有针对性地排查。
