# 本地知识库检索与智能体证据接口

此分支把已有动态 embedding、BGE reranker 和证据检索接口迁移到 InfiniCore 固定的 SDK 组合。Encoder、CLS 提取和分类头通过 InfiniTensor 调用 InfiniOps，在选定设备执行。CPU 负责分词、shape 元数据、归一化、索引排序及证据存储。没有模型前向回退。

合并顺序为 InfiniOps `feat/retrieval-operator-coverage` → InfiniCore `chore/update-retrieval-operators` → InfiniTensor `build/use-infinicore` → 此分支。具体提交见 `dependencies/embedding-lock.json` 和 submodule gitlink。InfiniRT 版本保持不变，图接口仍归 InfiniRT；本组合使用普通执行，没有迁回旧 ATen 私有捕获接口。

## 安装和模型准备

按 [动态 embedding 安装说明](DYNAMIC_BGE_EMBEDDING_CN.md) 构建。`dependencies/embedding-ops.json` 包含 embedding、reranker 所需的算子，包括分类头的 Tanh/Addmm。SDK 自动生成绑定；本次没有在 Tensor 添加逐算子设备 dispatch。

Python 依赖可用 `pip install './pyinfinitensor[agent]' --no-deps` 安装本包，再按所在平台安装 transformers、safetensors 和 MCP 1.28.0。寒武纪和昇腾保留各自匹配的 PyTorch/厂商插件，不能用通用 CUDA wheel 覆盖。MCP 需要 Python 3.10 以上。

昇腾启动前设置 `TASK_QUEUE_ENABLE=0` 和 `INFINIOPS_PROVIDER_MODULES=torch_npu`；寒武纪设置 `INFINIOPS_PROVIDER_MODULES=torch_mlu`。设置须在导入厂商插件前完成。

Embedding 模型准备沿用 `scripts/export_bge_small_zh.py`。Reranker 使用已下载的 `BAAI/bge-reranker-base` revision `2cfc18c9415c912f9d8155881c133215df768a70`：

```bash
python scripts/export_bge_reranker.py --source /path/to/verified-reranker-snapshot \
  --revision 2cfc18c9415c912f9d8155881c133215df768a70 \
  --output-dir models/bge-reranker --validate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

导出执行 PyTorch/ORT 对照并保存文件摘要；部署时只加载本地包。模型包较大，接口不会自动下载权重。

## 两阶段检索

准备每行含唯一 `id` 和 `text` 的 UTF-8 JSONL。文档先编码，查询时向量召回后对候选逐对打分：

```bash
python examples/python/knowledge_search.py --embedding-model models/bge-small-zh \
  --device nvidia build --documents documents.jsonl --index indexes/docs
python examples/python/knowledge_search.py --embedding-model models/bge-small-zh \
  --device nvidia query --index indexes/docs --reranker-model models/bge-reranker \
  --text '动态输入如何处理？' --recall-k 50 --top-k 5
```

`RerankerSession.score([(query, passage), ...])` 返回原始 FP32 relevance logits。按真实长度分组，受 batch 与 token budget 约束，仅 padding 到当前批最长长度，输出恢复到输入顺序。默认每批至多 8 对、最长 512 token。`rerank()` 分数降序，同分按文档 ID 稳定排序。分数不是概率；精排效果需在实际业务数据评估。

## 给智能体提供可核查的证据

`infinitensor_evidence` 处理版本化技术文档：保存原文区间和来源 revision；按宿主指定的 namespace、reader、平台和组件版本过滤；组合 BM25 与向量召回，按需精排；返回有预算的上下文与可再次读取的 citation。适用于智能体反复查询同一代码库/知识库时的证据查找、版本过滤和上下文控制。它不生成答案，也不把相关性当作任务成功。

证据 JSONL 每条至少含 `id, namespace, revision, source_uri, title, text, readers`。`readers` 是宿主提供的身份列表或 `['*']`；`platforms`、`versions` 为可选精确过滤字段。入库会保留原文偏移，并通过两个 tokenizer 的预算切块。索引构建与撤回由管理命令执行，MCP 只暴露 `search_evidence` 和 `read_evidence`。

```bash
python -m infinitensor_evidence build --store evidence.db --documents evidence.jsonl \
  --principal developer --namespace project --embedding-model models/bge-small-zh \
  --reranker-model models/bge-reranker --device nvidia
python -m infinitensor_evidence serve --store evidence.db \
  --principal developer --namespace project --embedding-model models/bge-small-zh \
  --reranker-model models/bge-reranker --device nvidia --rerank
```

服务单工作线程持有模型，不并发操作单模型实例。重复查询和精排 pair 使用有界精确输入缓存；更新或撤回文档后，旧 citation 会标为 superseded 或不可读。默认混合召回，`--rerank` 显式启用精排；`--lexical-only` 是独立的 CPU 基线选项，不是执行失败时的隐式回退。服务 stdout 留给 MCP 协议。

## 迁移回归

```bash
python -m unittest discover -s pyinfinitensor/tests -p 'test_evidence_*.py' -v
python -m unittest discover -s pyinfinitensor/tests -p test_reranking.py -v
python scripts/validate_retrieval_integration.py --embedding-model models/bge-small-zh \
  --reranker-model models/bge-reranker --device nvidia \
  --documents evidence.jsonl --random-shapes 100 --output results/retrieval-migration
```

真实设备脚本比较同设备 PyTorch FP32：embedding 演示序列、35 组矩阵、100 组随机形状及边界；reranker 变长、空字符串、512 截断和跨批次顺序；检查实际 provider 实例、权重驻留、稳定 citation 与重复查询。该脚本是本次迁移回归，不替代六端点 M1 长稳、OOM、干净安装和性能验收，也不宣称业务检索质量提升。
