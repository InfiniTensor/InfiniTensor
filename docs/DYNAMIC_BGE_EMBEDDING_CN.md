# 使用新版 InfiniOps 执行动态中文 Embedding

本改动为普通执行路径接入 `BAAI/bge-small-zh-v1.5`。同一模型实例保留一份计算图与权重，连续接受不同 batch/sequence 的输入，返回 FP32、512 维的归一化向量。文本接口认证范围为每批最多 32 条、最长 512 token；更长列表顺序分批，按每批实际最长长度右侧 padding。

目标平台为 Linux x86-64 + NVIDIA 单卡。其他设备沿用现有 runtime 入口，本改动不构成其他平台认证。没有加入图捕获、算子融合、并发服务、Reranker 或智能体协议接口。

## 固定依赖与安装

`dependencies/embedding-lock.json` 记录源码和模型版本，`3rd-party/InfiniOps` / `InfiniRT` gitlink 与锁文件一致。InfiniOps 固定到已合入官方 master 的 [PR #1009](https://github.com/InfiniTensor/InfiniOps/pull/1009)，提交为 `640475a658de1f26bbf7c018fd0e0f8d5303a74c`，不再依赖开发分支。

构建仍使用外部安装前缀，不把 InfiniOps 或 InfiniRT 源码编入 InfiniTensor。以下示例在仓库根目录执行；编译器、驱动和 CUDA Toolkit 须与 PyTorch 匹配。

```bash
git submodule update --init --recursive
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install pybind11 transformers==4.44.2 onnx==1.17.0 onnxruntime-gpu==1.20.2 onnxsim safetensors

cmake -S 3rd-party/InfiniRT -B build-rt \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/prefix-rt" \
  -DWITH_CPU=ON -DWITH_NVIDIA=ON
cmake --build build-rt -j2
cmake --install build-rt

cmake -S 3rd-party/InfiniOps -B build-ops \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/prefix-ops" \
  -DWITH_TORCH=ON -DWITH_CPU=ON -DWITH_NVIDIA=ON \
  -DINFINI_RT_ROOT="$PWD/prefix-rt" \
  -DINFINI_OPS_OPS="$PWD/dependencies/embedding-ops.json"
cmake --build build-ops -j2
cmake --install build-ops

make install-python INFINI=ON \
  INFINIOPS_ROOT="$PWD/prefix-ops" INFINIRT_ROOT="$PWD/prefix-rt" \
  PIP_INSTALL_ARGS=--no-deps \
  PYTHON="$(command -v python)"
```

`embedding-ops.json` 是本模型所需的选择构建配置，不是完整 InfiniOps 算子集，也不代表其他 ONNX 模型都能执行。普通构建不再强制要求旧的 capture-memory/value-only-pooling 私有辅助头；旧辅助头存在时保留既有兼容路径，缺少 capture-memory 时不启用原有 ATen 图捕获入口。新的图执行适配留给后续 PR，不能将这些 provider 宣称为 capture-safe。

## 准备模型与断网运行

联网准备阶段按固定 revision 导出一个动态 opset-17 模型，ONNX 输出为 GPU 上提取的 CLS 向量。导出包含 PyTorch/ORT 一致性检查，并为权重、tokenizer 和 ONNX 文件生成 manifest：

```bash
python scripts/export_bge_small_zh.py \
  --revision 7999e1d3359715c523056ef9478215996d62a620 \
  --output-dir models/bge-small-zh --validate
```

模型包也保存参考 safetensors，以便离线验证。推理阶段不使用 PyTorch/ORT 前向回退；ATen 作为 InfiniOps 的数值 provider 使用。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python examples/python/bge_embedding.py --model-dir models/bge-small-zh \
  --query '如何执行动态中文向量推理？' \
  --document 'InfiniTensor 接受不同长度文本并生成向量。' \
  --document '该文档描述图像分类数据集。'
```

```python
from pyinfinitensor.embedding import EmbeddingSession

session = EmbeddingSession("models/bge-small-zh", device="nvidia", device_id=0)
passages = session.encode(["中文文档", "包含 English 的更长文档"])
queries = session.encode(["查找相关文档"], kind="query")
assert passages.shape == (2, 512)
assert queries.shape == (1, 512)
```

查询模式添加 `为这个句子生成表示以用于检索相关文章：`，文档模式不添加。截断长度包含该前缀和特殊 token。CPU 执行 tokenizer 和最终 L2 归一化，Encoder 与 CLS 提取在设备执行。精度上下文在调用期间关闭 provider 的缩减精度开关，退出后恢复调用者设置；M1 为单实例串行使用，不提供跨线程精度状态隔离。

## 动态接口和生命周期

`OnnxStub.prepare(input_shapes)` 按输入名准备具体形状；旧的 `set_input(list_of_shapes)` 转到相同路径。`run_numpy(inputs)` 校验输入名、dtype、rank 和共享符号维度，再准备、复制、执行，返回独立拥有内存的 NumPy 输出。未准备的动态模型调用底层 `run()` 会报错。

维度来源保存在模型模板中；导入时的临时值 1 不会成为永久形状。Shape 子图只接受常量和张量维度，普通词嵌入 Gather 等数据节点留在设备。不支持的数据依赖 shape 路径明确报错，不回读中间激活模拟执行。

Reshape/Expand/Slice 参数随准备更新。Slice 保存原始端点，并在拓扑形状推导时按当前输入重新归一化，避免初始小形状永久夹紧端点。Gather 形状推导只检查元数据，不从旧设备分配读取索引。初始化上传权重一次；动态变化复用 master 已有 LazyAllocator，不引入新的分配器或多形状图缓存。

`weight_upload_count`、`weight_upload_bytes`、`weight_addresses` 用于检查权重驻留。普通参数校验失败保留先前可执行状态；更新失败会尝试恢复原形状，恢复失败则标记未准备并拒绝继续执行，原始异常向调用者传播。

## 验证和初步性能基线

```bash
python -m unittest discover -s pyinfinitensor/tests -p test_onnxstub.py -v
python -m unittest discover -s pyinfinitensor/tests -p test_embedding.py -v
python scripts/validate_bge_embedding.py --model-dir models/bge-small-zh \
  --output results/bge --random-shapes 100 --benchmark-iterations 50
```

验证入口检查包内摘要，先比较 PyTorch 与 ORT，再比较 InfiniTensor；覆盖演示序列、35 组固定形状、S=1/B=32 边界、100 组固定种子随机形状及文本分批/前缀/截断/顺序。规范化向量要求 `atol=1e-4, rtol=1e-3`、余弦不低于 `0.99999`、范数误差不超过 `1e-5`。同时检查权重计数/字节/地址和旧输出内容。

ORT 使用 CUDA I/O binding，并单独保存 profiling 的逐节点 provider 分配。数据计算节点不得通过 CPU fallback 验证；CPU shape 元数据节点与设备前向分开核查。InfiniOps 自带 trace 可独立取证，避免将日志开销计入基线：

```bash
mkdir -p results
INFINI_OPS_TRACE_CALLS=1 python scripts/validate_bge_embedding.py \
  --model-dir models/bge-small-zh --output results/trace --trace-only \
  2> results/infiniops-trace.log
```

性能部分是共享设备、单进程内的初步基线：固定五组 B/S，各引擎预热 20 次，保存测量原始样本和 p50/p95。计时包括输入复制、host 提交、同步、结果回读和归一化，不包括 tokenizer；不冒充纯 kernel 时间。它不替代三轮独立进程、轮换顺序及长稳等完整 M1 验收，不设性能领先门槛。
