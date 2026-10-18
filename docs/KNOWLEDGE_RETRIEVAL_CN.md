# 中文知识库召回与精排

本场景把现有 BGE embedding 接口扩展成可离线运行的两阶段检索：文档预先向量化，查询通过向量点积召回，再使用 `BAAI/bge-reranker-base` 对查询与候选文档联合编码，返回相关性排序。它适合作为本地知识库或 RAG 系统的检索层；不包含答案生成服务。

## 模型与执行边界

- 召回：`BAAI/bge-small-zh-v1.5`，512 维 CLS 向量、L2 归一化，查询添加既有中文检索指令。
- 精排：`BAAI/bge-reranker-base`，12 层、隐藏维度 768 的 XLM-R 跨编码器。直接输入原始查询/文档文本对，不添加 embedding 查询指令；输出未经 sigmoid 的相关性 logit，分数越大越相关。
- 固定版本和 SHA256 见 `dependencies/retrieval-lock.json`。精排模型以 eager attention、opset 17、FP32 导出，模型包包含 tokenizer、ONNX 和独立参考权重。导出时先验证 canonical PyTorch、显式 position IDs 包装及 ORT CPU 的一致性。
- tokenizer、XLM-R position IDs、组批和最终排序在 CPU 执行。整个 encoder 与分类头通过 InfiniTensor → InfiniOps ATen provider → InfiniRT 在指定加速卡执行，没有 PyTorch/ORT 前向回退。
- 一份动态计算图、一份权重。按批次实际最大长度右侧 padding，最大 512 token，保留原始输入顺序。默认最大 batch 8；本场景设备测试范围为 B=1..8、S=4..512，API 可以另配更大容量，但应追加资源验证。
- FP32 策略在接口内禁用 CUDA/MLU TF32 和 NPU HF32，结束后恢复外部设置。模型及分词器文件一起校验，索引记录整个推理/分词包身份，拒绝不兼容的查询模型。

## 实现优化

1. `batching="length"` 根据实际 tokenizer 长度稳定排序，以 batch 上限和 padded token 预算组批，并将分数还原为原始顺序；没有有限长度桶或静态模型副本。
2. `memory_views=True` 让同字节数的中间 Reshape/Flatten/Identity/Squeeze/Unsqueeze 共享存储。分配器按别名组的全部消费者计算生命周期；公共输入、输出与权重保持独立存储。`view_ops` 和 `view_copy_bytes_elided` 表示当前形状的静态消除量，不是实测显存带宽或执行耗时。
3. `fuse_linear_bias=True` 将满足约束的 MatMul+常量向量 Add 合为带 bias 的 Matmul，再通过 InfiniOps Addmm 执行。只折叠稠密投影的前导行维度；多头 attention 的批次广播路径保持原有语义。
4. `execution_mode="auto"` 复用已有 InfiniRT 图入口，连续同形状达到阈值后捕获。NVIDIA、寒武纪和昇腾走统一接口。变长候选经常改变形状时可能没有收益，不能把固定形状 graph 加速比外推到完整检索请求。

旧 `OnnxStub` 和 `EmbeddingSession` 的上述两个新优化默认关闭。新 `RerankerSession` 默认启用它们。接口与示例程序默认选择 normal，避免变长请求反复捕获造成延迟回退；固定形状的持续请求可以显式选 auto/graph。可关闭开关做同代码性能消融。

## 构建与模型准备

沿用 `docs` 中的 M1 Linux 构建流程及 `dependencies/m1-lock.json`。`scripts/install_m1_clean.sh` 的算子选择新增 `addmm`、`tanh`（实现 8）。只升级 InfiniTensor 而复用缺少这两项的旧 InfiniOps 二进制不够。InfiniOps 另修复了纯原生昇腾图中的 allocator 初始化：预热后未使用 ATen 的图不建立 provider 私有池，避免只初始化 caching allocator、未初始化 workspace allocator 时销毁崩溃；补丁和锁文件同步更新。InfiniRT 无新增源码修改。

```bash
python -m pip install -e './pyinfinitensor[retrieval]'
python scripts/export_bge_reranker.py --source /models/verified-reranker-source \
  --revision 2cfc18c9415c912f9d8155881c133215df768a70 \
  --output-dir /models/bge-reranker-base --validate
```

`--source` 必须是事先下载并核验的完整本地模型。revision 字符串用于记录来源，不能替代下载文件的校验。锁文件给出的 ONNX 摘要对应本次固定导出工具链；其它工具链重导出后必须重新验证并冻结新 manifest，不能伪造为相同文件。

## 离线使用

文档文件每行包含唯一字符串 `id` 和字符串 `text`：

```json
{"id":"install","text":"安装 Python 后，可以使用 pip 安装所需依赖。"}
{"id":"search","text":"向量检索根据文本的语义相似度查找文档。"}
```

模型准备后设置离线开关；索引和查询均不需要网络：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python examples/python/knowledge_search.py build \
  --embedding-model /models/bge-small-zh --documents documents.jsonl \
  --index knowledge.npz --device nvidia
python examples/python/knowledge_search.py query \
  --embedding-model /models/bge-small-zh --reranker-model /models/bge-reranker-base \
  --index knowledge.npz --query '如何安装 Python？' --recall-k 50 --top-k 10
```

使用 `--device cambricon` 或 `--device ascend` 选择其它后端，同时按既有 M1 文档配置厂商 SDK、动态库及 provider。省略 `--query` 时从 stdin 逐行读取 `{"id":"q1","query":"问题"}`，两份模型只加载一次，stdout 每行返回一个结果对象。索引 NPZ 自包含文档、ID、向量与模型身份；加载禁用 pickle。相同分数按文档 ID 稳定排序。单实例串行处理，不包含并发调度。

Python 接口：

```python
from pyinfinitensor.reranking import RerankerSession
session = RerankerSession('/models/bge-reranker-base', device='nvidia')
scores = session.score([('如何安装 Python？', '使用系统包管理器安装 Python。')])
results = session.rerank('如何安装 Python？', ['使用包管理器安装。', '熊猫生活在中国。'])
```

## 可复现验证与性能解释

```bash
python scripts/prepare_reranking_dataset.py --parquet /data/dev.parquet --output /data/subset
COMMON="--model /models/bge-reranker-base --embedding-model /models/bge-small-zh --dataset /data/subset --device nvidia"
python scripts/accept_retrieval.py --stage accuracy $COMMON --output results/accuracy
python scripts/accept_retrieval.py --stage quality $COMMON --output results/quality
python scripts/accept_retrieval.py --stage stress $COMMON --output results/stress
python scripts/accept_retrieval.py --stage benchmark $COMMON \
  --requests results/quality/performance-requests.json --output results/benchmark
```

验证分别记录失败状态，任何异常返回非零。accuracy 包括 126 组固定/随机动态形状，每组更换内容再执行以验证实际 replay、权重驻留和旧输出所有权，以及混合长度文本和 embedding 回归。stress 默认 1000 次实际动态请求，属于新路径压力回归，不替代原 M1 六端点两小时验收。

固定应用子集按事先定义的规则从 T2Reranking 选 64 个有正例的查询及 992 篇去重文档；记录原始 ID、文本与 SHA256，不能作为完整榜单成绩。报告 embedding 召回和 reranker 的 NDCG@10、MRR@10、Recall@10/50，并同时检查同设备 PyTorch 的 logit 精度与 Top-10 一致性。模型精度通过并不保证精排在所有业务指标上提升，应保留质量下降的真实结果。

benchmark 依次对比同一份当前源码的 baseline（新优化关闭）、engine（内存共享/bias 融合）、packed（再加长度组批）、graph（再加 auto），以及相同 FP32 eager PyTorch 的输入顺序/长度组批两种路径。每轮独立进程，20 次预热、100 次测量，三轮轮换引擎顺序；同一张卡串行执行。固定张量、实际 Top-50 召回请求流以及交错短长文档的 padding 压力场景分开报告。baseline 的分类头仍使用本版本已有的带 bias Gemm 路径，因此它不是历史发布二进制的绝对对照。

计时采用设备同步边界，包含 host 提交及输出传输；完整候选请求另外包含 tokenizer、组批及还原开销，不冒充纯 kernel 时间。原始样本、分段耗时、有效/padding token 数、显存/内存和共享负载信息一起保存。这里的速度只描述共享环境，不要求停止其他任务，也不宣称独占硬件性能。

向量索引目前使用 CPU 全量点积，适合中小知识库；百万级文档需要外接 ANN 索引并单独评估召回损失。FP16/BF16、INT8、融合 attention、多形状图缓存及服务并发不属于本次交付。
