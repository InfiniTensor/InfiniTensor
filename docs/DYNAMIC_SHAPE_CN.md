# 动态 Shape 子图编译与执行

一个模型的形状不一定在导出时就定下来。批量大小随请求变化，图片尺寸随来源变化，序列长度随输入变化。这类模型导出成 ONNX 时，变化的那些维度被写成一个名字而不是一个数字，真正的尺寸要等到推理时才知道。

本文说明如何导入这样的模型、如何在同一个实例上连续换形状推理、以及部署固定形状时可以省下什么。

## 目录

- [快速开始](#快速开始)
- [动态维度是怎么表示的](#动态维度是怎么表示的)
- [运行时目标 Shape](#运行时目标-shape)
- [连续换形状](#连续换形状)
- [换形状要花多少内存](#换形状要花多少内存)
- [把维度钉住](#把维度钉住)
- [支持范围](#支持范围)
- [已知限制](#已知限制)
- [复现](#复现)

## 快速开始

编译并安装 python 前端：

```bash
make build
make install-python
```

跑演示程序，它自带三个动态 Shape 模型，不需要外部模型文件：

```bash
python examples/python/dynamic_shape_inference.py
```

每个模型会连续推理五次，每次换一组形状，并把结果与 onnxruntime 对比：

```
 batch [batch=1]: y has shape [1, 2, 3], differs from onnxruntime by at most 0.00e+00
 batch [batch=4]: y has shape [4, 2, 3], differs from onnxruntime by at most 2.98e-08
 image [batch=1, height=8, width=8]: y has shape [1, 4, 4, 4], differs from onnxruntime by at most 1.04e-07
sequence [batch=2, length=7]: y has shape [2, 7, 32], differs from onnxruntime by at most 3.58e-07
```

两侧都在 CPU 上用 float32 求同一张图，但求和的累加顺序是各自 kernel 的事，所以结果只在舍入范围内一致，而不是逐位相同。

也可以指定自己的模型和形状，`名字=尺寸` 按维度名给，单个数字给所有动态维度：

```bash
python examples/python/dynamic_shape_inference.py model.onnx batch=1 batch=8
python examples/python/dynamic_shape_inference.py model.onnx 1 4 16
```

## 动态维度是怎么表示的

ONNX 的每一个维度要么带一个数字，要么带一个名字。带名字的那些是导出时没有定下来的，导入后逐维记在张量上：

```python
from pyinfinitensor.onnx import OnnxStub
from pyinfinitensor import backend
import onnx

stub = OnnxStub(onnx.load("model.onnx"), backend.cpu_runtime())
x = stub.inputs["x"]
for axis, desc in enumerate(x.dim_descs()):
    print(axis, "dynamic" if desc.dynamic else "fixed", desc.name)
```

带数字的维度是**固定的**，不能改。给它另一个尺寸会被拒绝，而不是默默算错：

```python
stub.set_input([[1, 3, 224, 224]])   # 可以，batch 是动态的
stub.set_input([[1, 8, 224, 224]])   # 报错，通道数是模型写死的 3
```

这条拒绝是后面所有优化的依据：一个不可能改变的维度，凡是只由它推出来的东西也不会改变。

## 运行时目标 Shape

`Reshape` 的目标形状不要求在导入时就能算出来。导出器通常把它拼成一条子图——读出输入形状，取出其中几个维度，算一算，再拼回一个新形状：

```
Shape → Gather → Unsqueeze → Concat → Reshape
```

这条链在导入时不需要求值。`set_input` 给出真实形状后，形状推断会沿着它走一遍，`Reshape` 读到的是这一次的目标，而不是导入时的那个。演示程序里的 `sequence` 模型就是完整的一条：

```bash
python examples/python/dynamic_shape_inference.py sequence
```

参与这条链的算子有 `Shape`、`Gather`、`Unsqueeze`、`Squeeze`、`Slice`、`Concat`、`Identity`，以及维度上的算术 `Add`、`Sub`、`Mul`、`Div`、`Max`、`Min`、`FloorDiv`、`FloorMod`。

## 连续换形状

同一个实例可以连续换形状推理，不需要重新加载模型：

```python
for shape in ([1, 3, 8, 8], [2, 3, 8, 8], [8, 3, 12, 12], [3, 3, 4, 4], [1, 3, 8, 8]):
    stub.set_input([shape])
    stub.inputs["x"].copyin_numpy(data_of(shape))
    stub.run()
    print(stub.outputs["y"].shape())
```

`set_input` 会重新推断所有下游形状并重新规划内存。形状与上一次相同时整段跳过，因为重新排布内存比跑一遍图还贵，而一个已经被接受过的形状不需要再验一遍。

## 换形状要花多少内存

重新排布不等于重新分配。activation 的存储量停在已见过形状的最高水位上，后来一个装得下的形状直接复用，不再向 runtime 要内存；超出水位时按 1.5 倍增长。所以一段来回变化的形状序列，稳态下一次分配都不做。

```python
stub.activation_allocations()   # 累计向 runtime 要过几次
stub.activation_peak()          # 当前形状需要多少字节
stub.activation_capacity()      # 实际持有多少字节，即水位
stub.allocated_bytes()          # 连权重和堆一起，一共持有多少
```

`peak` 与 `capacity` 的差就是换来复用的那部分余量。`trim_memory()` 把余量还掉，代价是下一次换形状要重新分配。

`examples/python/dynamic_shape_benchmark.py` 把这件事量出来，同时对比「复用」与「每次推理后 trim」两种策略：

```bash
python examples/python/dynamic_shape_benchmark.py model.onnx \
    --sweep batch=1,1,1,1 height=32,128,256,32 width=32,128,256,32
```

以下为早期版本在本机 CPU、两个自带模型上的历史测量，不能直接用作当前修改的性能结论。分配次数和持有字节是数出来的；时间跨运行有约五成的漂移，因此报告复用相对精确分配的倍率：

| 模型 | 序列 | 分配次数（复用 / 精确） | 平均持有（复用 / 精确） | 每次推理 |
|---|---|---|---|---|
| 17 算子图像模型 | 32/128/256/32 循环 4 轮 | 0 / 12 | 5.00 MiB / 1.60 MiB | 快 11%–17% |
| 51 算子 attention | seq 8/16/32 循环 6 轮 | 0 / 36 | 0.07 MiB / 0.04 MiB | 快约 5%，在波动内 |

三点要说准：

- **分配次数被完全消掉**，稳态下是 0 而不是「变少」。
- **省下的时间落在排布那一段，不在算子执行上。** 图像模型上 12 轮交替试验测得复用的 `layout` 是精确分配的 0.42 倍（逐轮 0.37–0.48），而 `run` 的比值是 1.03，即执行时间不变。收益随张量大小放大：attention 的张量只有 0.07 MiB，分配成本相对 51 个算子的执行微不足道，脚本会自己标注差异落在试验波动内。
- **代价是多占内存**：图像模型平均持有 5.00 MiB 对 1.60 MiB，多三倍。全序列峰值两边一样，因为序列里含最大的那个形状，两边在那一刻都得持有它；差别在于之后还持不持着。

时间不要用单次总时长比较——总和被少数几次慢推理决定，而它们落在哪个策略上取决于谁先跑。脚本默认交替测 5 轮取中位试验，就是为了这个：单轮各测一次时，换个顺序结论会反过来。

## 把维度钉住

导出时留成动态、部署时只服务一个尺寸，是很常见的情况。这件事简化器不可能知道——它在部署存在之前就跑完了。告诉框架之后，整条只依赖这个维度的形状子图可以只算一次：

```python
stub.pin_dims("x", [0])          # 第 0 维从此固定在当前尺寸
dropped = stub.fold_shape_subgraph()
print(f"{dropped} 个算子被折叠掉了")
```

演示程序的 `--pin` 会对每个模型做这件事：

```bash
python examples/python/dynamic_shape_inference.py --pin
```

```
   image: pinned at this shape, 9 of those 9 shape operators became the same under every remaining inference; 0 remain
sequence: pinned at this shape, 7 of those 7 shape operators became the same under every remaining inference; 0 remain
```

`--fold` 则不钉住，只报告当前能折叠多少：

```bash
python examples/python/dynamic_shape_inference.py --fold
```

```
   image: the shape subgraph holds 9 operators, 0 of which are the same under every shape this model may be given and were worked out once; 9 remain to be computed per inference
sequence: the shape subgraph holds 7 operators, 0 of which are the same under every shape this model may be given and were worked out once; 7 remain to be computed per inference
```

在普通导入的模型上这个数字通常是 **0**。这不是缺陷：前端会先对模型做一遍简化，而简化器已经把「由 ONNX 声明的形状能推出来的部分」折完了。两者的知识来源相同，先跑的那个把它拿走。折叠能做的是简化器拿不到的那部分——部署钉住的维度。

钉住之后那个维度不能再改：

```python
stub.pin_dims("x", [0])
stub.set_input([[2, 3, 8, 8]])   # 报错，第 0 维已经钉在 1 上
```

这条拒绝是必须的。折叠的结果被写死进图里，钉住的维度事后改变会留下一个过期的答案，而过期的答案不会报错，只会算错。

## 支持范围

| | 支持情况 |
|---|---|
| 后端 | CPU（`backend.cpu_runtime()`）|
| 动态维度数量 | 不限，可以多个维度同时独立变化 |
| `Reshape` 目标 | 静态属性，或运行时张量 |
| `Expand` 目标 | 形状推导接受静态属性或运行时张量；本项目尚未提供 CPU Expand kernel，不声明 CPU 端到端支持 |
| `Tile` 次数 | 静态属性，或运行时张量；CPU kernel 已实现 |
| 几何算子（`Conv` `Pool` `ConvTranspose` 等）| 支持动态空间维度；构造期占位不足时自动保护，可用 `input_shapes=` 直接给出部署形状 |
| `Resize` | 按 `scales` 与按 `sizes` 均支持动态输入；`stretch` / `not_larger` / `not_smaller` 三种策略每次按当前输入重算 |
| `-1` 与 `0` 占位符 | 支持 |
| 形状子图算子 | `Shape` `Gather` `Unsqueeze` `Squeeze` `Slice` `Concat` `Identity`，及维度算术 |
| `Slice` 边界 | 常量或推导期可求值的 starts/ends 张量；axes/steps 需常量；CPU 支持正步长与空输出 |

## 已知限制

**形状变化只能从图的输入进入。** 换形状的唯一入口是 `set_input`。

**固定性已按维度传播，未提供精确来源的算子保守回退。** `dimSources()` 描述每个输出维依赖哪些输入维，`spreadFixedDims()` 据此传播维度描述；整数 shape value 另有逐元素固定性。动态选择边界会使 Slice 结果保持可变，即使这次选中的数据元素本身是固定值。

**混合固定/动态的形状张量保持完整。** 折叠只删除输出已完全固定的算子，以及因此不再使用的形状计算。读取固定元素的 Gather/Slice 可以折叠，无需拆开其仍被动态消费者使用的输入。`fold_shape_subgraph()` 返回实际删除节点数；重复调用不会持续插入 Slice/Concat。图输出保留生产者和外部引用。

**Gather 接受 ONNX 的合法负索引。** Int32/Int64 的合法范围为 `[-axis_length, axis_length-1]`。形状推导优先使用当前 shape value，避免读取上一轮执行所得的索引缓冲区；CPU kernel 在执行时检查运行时索引是否越界。

**Slice 的边界按当前维度裁剪。** int64 极值哨兵在裁剪前保持原值，过度负边界也会裁剪；未切维随输入大小更新。空切片从首次导入起保持空，不被动态占位保护补成一个元素。CPU 负步长仍明确报不支持；形状推导能描述逆向范围，不代表已有对应 CPU 执行能力。反向 ONNX 导出 Slice 仍未实现。

**几何算子的构造期占位保护。** 算子在构造时就要推出输出形状，而此时符号维度只有占位的 1。这个 1 常常小于算子的几何需要：3×3 卷积在 1×1 上按 ONNX 的公式算出来是 -1，直接构造会因为负维度中止。所以构造期对声明了动态维度的输入做一次保护，把非正的推导结果抬到 1，等 `set_input` 给出真实形状后再重新推断出正确的几何。这只在输入声明了动态维度时发生：完全静态的图里负维度仍然是错误，仍然中止，不会被这层保护掩盖。

保护让图能建起来，但它抬上去的 1 并不是模型的真实形状。如果导入期就知道会部署在哪个形状上，可以直接把它交给导入：

```python
stub = OnnxStub(model, backend.cpu_runtime(), input_shapes={"x": [1, 3, 224, 224]})
```

这些形状同样走 `DimDesc` 校验，改动固定维度会被拒绝，动态维度之后仍然可以用 `set_input` 任意变化。

需要注意池化的偏离：窗口大于输入时，ONNX Runtime 会拒绝负的几何、并对零长度返回空张量，本项目一律给到 1。这是有意的——本项目的形状链会把 0 带进 `Reshape` 的目标，而 ONNX 里目标的 0 表示「保持输入这一维」，含义正好相反。

**按 `sizes` 的 `Resize` 每次重算比值。** 内部按比值缩放，而比值是「请求的尺寸 ÷ 输入对应维」。这个比值只描述取它时的那个输入形状，所以不能在构造期算一次就留着：动态维在构造期只有占位的 1，拿它做除数得到的比值会把之后每个形状都按这个倍数放大，而不是缩放到模型要求的尺寸。请求的尺寸是模型定下的，比值跟着输入走，因此保留前者、每次形状推断时重算后者。`not_larger` 与 `not_smaller` 这类跨轴共用一个比值的策略同样每次重选，所以共用的那个比值也随输入变化。

相应地，被 `sizes` 钉住的轴在固定性传播里不跟随任何输入维——它就是模型要求的那个数，输入怎么变都一样；按 `scales` 缩放的轴则是输入的倍数，跟随对应输入维。`not_larger` / `not_smaller` 下比值由所有参与轴共同决定，所以这些轴互相跟随。

**`Expand` 的目标可以是运行时张量，且按元素判断固定性。** 导出动态输入上的 `Expand` 时，目标通常写成对输入形状的计算（`Expand(x, Concat(Shape(x)[0:1], ...))`），模型里没有常量可读，所以目标接成图的一条边，每次形状推断重新读。

固定性按广播语义逐维判断：目标在某一维要求大于 1，则输出恒为那个数——广播要么与它一致，要么把 1 撑上去，输入怎么变都不改变结果，这一维不跟随任何输入维；目标要求 1，则该维保留输入自身的尺寸，跟随对应输入维。目标来自边时还多一层判断——只有那一位元素本身不会变（`isShapeValueFixed`）才作上述结论，会变的元素既可能钉住这一维也可能让位给输入，无法预先断言，退回保守答案。所以 `[N,1,4]` 配上算出来的 `[N,3,4]`，输出会正确描述成「第 0 维跟随输入，第 1、2 维固定」。

**`Tile` 的次数可以是运行时张量。** 与 `Expand` 同一个模式：次数通常写成对输入形状的计算，模型里没有常量可读，所以接成图的一条边每次重读。固定性的判断比 `Expand` 简单——输出的每一维都是输入对应维的固定倍数，倍数不变时这一维恰好随输入变化，所以跟随输入对应维；次数为 0 时该维恒为空，不跟随任何输入维。次数来自边时同样按元素判断，那一位会变则退回保守答案。

**符号维度参与 `Reshape` 目标的算术时无法导入。** 导入期用「每个动态维当作 1」求解 `Reshape` 目标，所以目标里含有对符号维的整除时，占位算术不成立，导入就会失败。例如输入声明为 `["batch", "length", "width"]`、目标是 `[batch, length, 4, width/4]` 时，`width` 占位为 1，`1/4 = 0`。把这个维度在 ONNX 里声明成数字可以绕过，这也是导出时通常的做法。

**固定性的传播需要一遍形状推断。** 一个张量要先有生产者，才能读到它从生产者那里继承了什么，所以这是一遍图上的 pass 而不是在每个算子加入时完成的。导入本身不跑形状推断，所以传播在第一次 `set_input` 或 `pin_dims` 时才生效。直接用 `addOp` 搭图时需要显式调用 `shape_infer()`。

## 复现

演示程序：

```bash
python examples/python/dynamic_shape_inference.py          # 三个模型各五组形状
python examples/python/dynamic_shape_inference.py --fold   # 报告可折叠的算子数
python examples/python/dynamic_shape_inference.py --pin    # 钉住维度后折叠
```

与 onnxruntime 的一致性测试，`rtol=1e-4`、`atol=1e-5`，逐个形状比对输出形状和数值：

```bash
python -m pytest pyinfinitensor/tests/test_dynamic_shape_ort.py -v
```

误差限的依据：两侧都在 CPU 上用 float32 求同一张图，但求法不同——求和的累加顺序是各自 kernel 的事——所以结果只在舍入范围内一致，而不是逐位相同。实测的不一致比这个限低约四个数量级，其中一个测试直接断言这个余量，以免这个限松到掩盖真实缺陷。

后端测试：

```bash
build/Release/test_shape_fold          # 形状值传播与折叠
build/Release/test_nativecpu_slice     # CPU slice kernel
build/Release/test_nativecpu_matmul    # CPU matmul kernel，含 bias 与 batch
build/Release/test_graph               # 含容量复用与分配计数的校准
```

内存与延迟测量，同时验证两种策略输出逐位一致：

```bash
python examples/python/dynamic_shape_benchmark.py model.onnx --sweep batch=1,2,4,8
```

需要 `onnxruntime` 才能跑一致性对比，没有装的话相关测试会跳过而不是失败：

```bash
pip install onnxruntime
```

真实 PyTorch 导出模型另有可复现生成器和正式回归：

```bash
python examples/python/dynamic_shape_attention.py real_attention.onnx
python -m pytest pyinfinitensor/tests/test_real_attention.py -v -s
```

此组还需要 `torch`；缺少 `torch` 或 `onnxruntime` 时会跳过，不能把跳过算作真实模型验收。生成器使用 hidden size 32、4 heads、seed=0、opset 18、`dynamo=False`，输入和输出的 batch/seq 均动态。`-s` 保留模型 SHA256、版本和逐形状误差。

测试在默认和关闭简化两条路径各用一个实例连续执行 `(2,5),(1,1),(2,7),(3,16),(1,3),(2,5)`。折叠另固定 seq=5，仅变化 batch 为 `2,1,8,3,2`，逐值比较 fold 前后结果，检查实际删除数、第二次 fold 不扩图及非法 seq 被拒绝后的恢复。

2026-09-08 夜间历史 Python 全集为 134 passed、2 skipped、13 subtests passed，4 项真实模型测试全部执行；两项跳过均要求 CUDA。模型已落盘且 SHA256 与归档相同，34 次 ORT 对比最大绝对误差 `1.7881393432617188e-07`。默认导入算子数 33→31，关闭简化 42→36，输出逐值不变。这是部分 pin 的功能结果；当前性能对照见下节。

## 2026-09-08 多输入拒绝恢复

多输入拒绝恢复的完整回归为 134 passed / 2 CUDA skipped / 13 subtests passed（8.60s）。该历史检查点C++未改；后续G1转置修复的重建与全集见下文。

后续输入不符合固定维、pin 或维度校验时，先前输入的形状会恢复；旧数据仍可执行。此保证限于 change_shape 拒绝，不覆盖任意 shape_infer 或内存分配异常。


## 2026-09-08 当前性能对照

导出真实 attention 后运行（Linux CPU，输出目录须新建）：

```bash
python3 examples/python/dynamic_shape_attention.py real_attention.onnx
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -B examples/python/dynamic_shape_comparison.py real_attention.onnx \
  --out results-new --trials 12 --repeat 20 --warmup 3
```

默认简化，seq=5/64 的 plain/fold 对照只变化 batch=2,1,8,3,2，两个策略均 pin 且复用容量。短/大动态序列 trim/reuse 对照均不 fold。trim 位于 set_input 后、copyin 前，计时包含其成本。程序先做 ORT 和策略逐值校验，再保存 samples.csv、trials.json、summary.json、metadata.json、validation.json。每轮新图、交替顺序，但共享进程/runtime；绑定单个逻辑 CPU，未独占机器。

本次 44 次 ORT 校验通过，策略间逐值相等，10,560 个计时样本完整。

| 场景（优化/基线） | 端到端配对比值中位数 [最小,最大] |
|---|---|
| fold_seq5 | 1.009 [0.876,1.122] |
| fold_seq64 | 0.997 [0.880,1.089] |
| capacity_short | 0.892 [0.727,1.016] |
| capacity_large | 0.990 [0.836,1.292] |

折叠没有稳定端到端加速证据。容量复用在每轮 120 次推理中将预热后激活分配从 100 次降至 0 次；平均持有内存短序列 72,336 对 33,552 bytes（2.156 倍），大序列 1,229,432 对 288,217.33 bytes（4.266 倍）。时延范围均跨 1，只报告本机观察，不主张稳定或普遍加速。

上述比值是逐 trial 配对后汇总，不是跨轮中位数相除。端到端从 set_input 前到输出复制完成，含 trim 和输入输出复制；不含导入、RNG、ORT 和计数器查询。分开统计变形和同形早退。内存为推理后 allocator 持有量，不是进程 RSS 或临时分配的瞬时峰值。

通用 `dynamic_shape_benchmark.py` 的 layout+run 不含数据复制；其总量及吞吐标签已明确这个范围，不能当作端到端。前文早期 0.42 倍数字仅为历史结果。

依赖限制：先导入 torch 再调用 onnxsim，进程退出可能出现 nanobind 引用计数提示；在不导入 InfiniTensor 的最小探针也可复现，exit 0。本项目不把 allocator 数据解释为进程无泄漏。详细原始证据随本地项目夜间日志归档，正式提交时另行附可公开材料。

## 标准ResNet18动态H/W验证

`examples/python/dynamic_shape_resnet.py`直接导出torchvision标准ResNet18，固定随机权重（weights=None）、eval、seed=20260908、opset18、dynamo=False。安装与torch匹配的torchvision；当前实测torch2.13.0搭配torchvision0.28.0。导出不依赖下载预训练权重，不作准确率主张。

```bash
python examples/python/dynamic_shape_resnet.py resnet18_dynamic.onnx
python -m pytest pyinfinitensor/tests/test_real_resnet.py pyinfinitensor/tests/test_gemm_transpose.py -q -s -ra
```

导入需`input_shapes={"images": [1, 3, 32, 32]}`给合法初始几何；同实例依次输入(1,3,32,32)、(2,3,40,48)、(1,3,64,48)、(3,3,32,64)、(1,3,48,32)、(1,3,32,32)。输出为[batch,1000]，测试同时验证内部Conv和GAP尺寸。默认及关闭简化共12次ORT对照，最大绝对误差2.592802047729492e-6（rtol=1e-4/atol=1e-5），返回初始尺寸逐值一致。模型无Shape Tensor子图，补充的是动态空间几何覆盖。

该模型末层Gemm的transB暴露CPU拒绝转置问题；NaiveMatmul现支持最后两轴transA/transB及组合，保留原批量广播边界。非方阵动态M三项最小回归改前失败、改后通过。2026-09-08该修复重建后C++51/51组（ShapeFold35/35），Python139 passed / 2 CUDA skipped / 4 exporter warnings / 13 subtests passed，12.02s。真实attention4项与ResNet2项均实际执行；缺依赖造成的skip不能作为真实模型验收。


## 双模型部署条件核对

| 模型 | 简化 | pin | 总节点前→后 | Shape节点前→后 |
|---|---|---|---|---|
| attention | True | dynamic | 33→33 | 12→12 |
| attention | True | partial | 33→31 | 12→10 |
| attention | True | fixed | 33→21 | 12→0 |
| attention | False | dynamic | 42→42 | 23→23 |
| attention | False | partial | 42→36 | 23→17 |
| attention | False | fixed | 42→19 | 23→0 |
| resnet18 | True | dynamic | 89→89 | 0→0 |
| resnet18 | True | partial | 89→89 | 0→0 |
| resnet18 | True | fixed | 89→89 | 0→0 |
| resnet18 | False | dynamic | 105→105 | 0→0 |
| resnet18 | False | partial | 105→105 | 0→0 |
| resnet18 | False | fixed | 105→105 | 0→0 |

上述是G1产物下既有折叠能力的12配置实测，均与ORT一致、fold前后逐值一致。ResNet不包含Shape子图。MatMul/Reshape固定性传播还有优化机会，尚未实现，不声称新增性能收益。

## MatMul维度固定性精化（2026-09-08）

MatMul新增逐轴dimSources：逻辑M只跟随A的对应轴，N只跟随B的对应轴；批量轴右对齐保留两侧来源，K与bias不决定输出尺寸。兼容transA/transB，当前等于1的动态批量轴仍保持动态。固定权重投影宽度不再被batch/seq污染。

相同attention ONNX、相同部署条件的11→18图谱比较：未简化/完全动态原来不删节点（42→42），现在删除7个（42→35）；未简化/seq=5 pin原来删6个（42→36），现在删13个（42→29），均额外删除7个。Shape子图分别23→16、23→10。默认简化路径已有常量优化，本次净删数仍0/2；全pin仍12/23。ResNet三种pin条件均0个Shape节点。

18双模型12配置全部通过ORT和fold前后逐值一致，重复fold为0，8个pin配置拒绝后恢复。新增固定性回归覆盖固定宽度折叠、动态M/K、四种转置、左右批量广播1→3→2→1。17局部6 passed/6 subtests；14重建后的19 C++51/51组（ShapeFold35/35），20 Python142 passed / 2 CUDA skipped / 4 exporter warnings / 19 subtests passed，12.74s，6项真实模型均运行。当前只证明减少形状节点，没有新增端到端提速主张。

全静态ONNX输入沿用可整体换形状的兼容约定，需pin_dims显式固定；维度固定性在shape_infer中传播。上述是图谱核对之后的新结果，先前表格保留为改前条件。动态Reshape和未简化CSE仍是后续候选。
