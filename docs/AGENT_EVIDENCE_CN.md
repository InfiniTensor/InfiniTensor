# 技术支持智能体的证据检索

本功能把已有的中文 embedding 和 reranker 封装为智能体可以调用的证据工具。
目标是处理真实的技术支持问题：错误码搜不到、不同硬件和依赖版本的说明混在一起、
上下文重复占用、文档更新后引用无法追溯。它负责提供可核对的来源，回答生成和工具执行仍由调用方完成。

实现位于独立 Python 包 `infinitensor_evidence`，不改变现有
`EmbeddingSession`、`RerankerSession`、`OnnxStub` 接口。本轮不修改 InfiniOps 或 InfiniRT。

## 数据与执行流程

1. 管理者将可信文档以 JSONL 导入，绑定 namespace、读者、平台、组件版本及不可变来源版本。
2. 使用两个模型 tokenizer 都能容纳的分块，保留原文位置；文档 embedding 由 InfiniTensor 执行并落盘。
3. 搜索先按调用者权限和环境筛选，再做 BM25、向量点积或 RRF 混合召回，可显式启用 cross encoder 精排。
4. 按预算选取完整片段，去除重复内容及同一来源的大段重叠；返回 context 和引用元数据。
5. `read_evidence` 按 citation ID 返回不可变片段，检查当前权限、撤回状态和环境范围，标记历史引用为 superseded。

模型前向只走 InfiniTensor；SQLite、FTS5、点积、tokenizer 和上下文拼装运行在 CPU。
这里采用精确向量扫描，没有 ANN 索引，适用于先验证实际知识库工作负载；大规模索引需另外评估。
没有调用 PyTorch/ORT 前向作为产品回退；验证脚本中的 PyTorch 仅用于独立数值对照。

## 安装与离线使用

使用已有验证过的 InfiniTensor 原生库、依赖锁和本地模型包，增加 Python extra：

```bash
python -m pip install './pyinfinitensor[agent]'
```

MCP 入口要求 Python 3.10+，固定 SDK `mcp==1.28.0`，以兼容已有 MLU/NPU 的 Python 3.10 环境。
这不是跟随 SDK 最新大版本的承诺。embedding 使用 BGE-small-zh-v1.5 模型包，
reranker 使用已有冻结的 BGE-reranker-base 模型包；模型准备方法见
[检索指南](KNOWLEDGE_RETRIEVAL_CN.md)。模型加载指定 `local_files_only=True`，运行期间不下载模型。
需要联网安装的依赖可先制作 wheelhouse 再离线安装。

每行文档示例（真实使用时替换为自己的来源、完整 commit 和组件版本）：

```json
{"id":"runtime/capture","namespace":"backend","title":"图捕获排查","text":"此处存储可信来源的完整原文。","source_uri":"https://git.example/docs/blob/<commit>/capture.md","revision":"<commit>","platforms":["ascend"],"versions":{"InfiniRT":["<runtime-sha>"]},"readers":["support-agent"]}
```

`id` 是 namespace 内的逻辑文档 ID；修改内容、来源、权限或版本都会形成新 revision hash。
`platforms` 可以是 `any/nvidia/cambricon/ascend/cpu`；`readers=["*"]` 表示该 namespace 内公开。
版本过滤采用组件版本字符串精确匹配，不做 semver 推断。文档未声明某组件表示通用说明，
声明 `*` 表示该组件通用；显式 `unknown` 不会自动匹配新的具体版本。

```bash
python -m infinitensor_evidence build --store /srv/evidence/backend.db \
  --documents /srv/evidence/documents.jsonl --namespace backend --principal support-agent \
  --embedding-model /srv/models/bge-small-zh --reranker-model /srv/models/bge-reranker \
  --device nvidia

python -m infinitensor_evidence search --store /srv/evidence/backend.db \
  --namespace backend --principal support-agent --platform ascend --version InfiniRT=<runtime-sha> \
  --query '图捕获失败应检查哪些运行环境？' --context-budget 2048 \
  --embedding-model /srv/models/bge-small-zh --reranker-model /srv/models/bge-reranker --device ascend

python -m infinitensor_evidence read --store /srv/evidence/backend.db \
  --namespace backend --principal support-agent --citation <citation-id> \
  --embedding-model /srv/models/bge-small-zh --reranker-model /srv/models/bge-reranker --device ascend
```

`--mode vector` 可单独使用向量召回；默认 `--mode hybrid` 不精排。
`--rerank` 是显式选项：通用 reranker 在真实项目术语上可能降低相关性，必须先评估。
`--lexical-only` 是明确标记的 CPU BM25 基线，不加载两个模型，也不是缺算子时的隐式回退。
仅词法索引不能原地变成完整向量索引，应另建数据库；修改分块策略也应重建。

默认预算按离线 reranker tokenizer 计数，并包含来源、版本和 citation 头部，
不包含外层 JSON 元数据。可用 `--context-tokenizer /srv/models/agent-llm` 指定实际智能体 tokenizer。
CPU 词法基线未指定 tokenizer 时使用 Unicode 字符数，并明确报告该单位。
没有足够预算容纳一个完整片段时返回 `context_budget_too_small`，不截出无法追溯的半段引用。

## MCP 接入

配置宿主启动独立 stdio 进程。下面的 `command` 和所有路径应替换为实际安装环境的绝对路径：

```json
{
  "mcpServers": {
    "backend-evidence": {
      "command": "/srv/venv/bin/python",
      "args": ["-m", "infinitensor_evidence", "serve",
        "--store", "/srv/evidence/backend.db",
        "--principal", "support-agent", "--namespace", "backend",
        "--platform", "nvidia", "--version", "InfiniRT=<runtime-sha>",
        "--embedding-model", "/srv/models/bge-small-zh",
        "--reranker-model", "/srv/models/bge-reranker", "--device", "nvidia"]
    }
  }
}
```

只暴露两个只读工具：

| 工具 | 输入 | 输出 |
|---|---|---|
| `search_evidence` | query、filters、top_k、recall_k、context_budget | 候选证据、完整 context、citation 列表、预算与分段时间 |
| `read_evidence` | citation_id、context_budget | 原始引用、版本、current/superseded 状态 |

宿主设置身份、namespace、固定平台/版本，模型不能通过参数冒充其他读者或扩大宿主环境范围。
这些是可信宿主配置，不是远程身份认证系统；不应让不可信客户端控制启动参数。
读取会同时核查当前文档和历史引用的平台/版本。来源 URI 只作为引用返回，不由工具任意打开。
单 worker 创建并串行使用两个模型，多个 MCP 请求可以同时到达，但不并发修改同一动态图。

候选相关性不等于问题已经可回答，原始 reranker logit 也不是正确率。
返回值明确说明 `answerability` 不由相关性推断。调用者应核对来源，证据不足时继续调查或转交人工。
检索内容可能含指令性文字，MCP instructions 提醒将其视为数据；这不是完整的提示注入防护系统。
目前验证覆盖真实 MCP 协议与模型执行，没有声称跑通某个大模型智能体的端到端问题解决率。

## 更新、一致性与缓存

同一 JSONL 再次 `build` 是增量 upsert，不上传未变文档的 embedding。
要把一个 namespace 整体替换为给定快照，显式添加 `--sync-namespace backend`：
不在输入中的该 namespace 文档会被撤回，其他 namespace 不受影响。空列表加该参数会撤回整个目标 namespace。
普通 upsert 不会因为文件中缺少某条记录而删除文档。

```bash
python -m infinitensor_evidence withdraw --store /srv/evidence/backend.db \
  --namespace backend --principal support-agent --id runtime/capture
```

管理命令必须由可信进程执行；不作为 MCP 写工具暴露。
更新前完成校验和编码，写事务原子提交；若另一进程在准备期间修改了输入文档，整批拒绝并提示重试。
搜索在同一读快照中完成过滤、排序和引用拼装。撤回和权限收紧也会禁止读取旧引用。
旧 revision 的数据为溯源保留，目前没有历史回收或数据库容量配额管理。

每个引擎实例缓存最多 128 个精确查询 embedding、4096 个精确 query/passage 分数，
还有限额内的 token 计数缓存。更新后的候选和权限始终重新从数据库读取，
修改文本产生新的分数缓存键；不是缓存最终回答，也不是语义近似命中。
缓存仅对完全重复输入有效，可用 `--no-cache` 对照。
当前即使不启用精排，也加载两个模型以复用分块、长度校验和预算策略；不宣称最低模型内存。

默认 focused query 最多 96 token，含标题的 passage 最多 384 token，均检查两个 tokenizer。
原文按最多约 300 token、重叠 40 token 切片；标题过长导致超限会明确拒绝。
source span 使用存储逻辑章节内的 Unicode 字符位置，不是整个文件的行号；
导入元数据另存原文件起始位置和不可变 permalink。

## 验证与版本合入

```bash
python -m unittest discover -s pyinfinitensor/tests -p 'test_evidence_*.py' -v
python scripts/prepare_agent_evidence.py --source /srv/source/InfiniTensor \
  --revision 4cf8f226486015e0d04b6d2e7043a8fc6a063f69 \
  --issues /srv/data/github-issues.json --output /srv/data/agent-pilot
python scripts/validate_agent_evidence.py --data /srv/data/agent-pilot \
  --embedding-model /srv/models/bge-small-zh --reranker-model /srv/models/bge-reranker \
  --device nvidia --output /srv/results/agent-pilot
```

仓库 `benchmarks/agent_evidence` 已保存本轮完整的文档、查询和 manifest，
可直接作为 `--data` 输入，不必重新访问 GitHub API。冻结文件保留原始字节，
由 `.gitattributes` 禁止换行转换，以保证跨平台 SHA256 一致。

真实问题集来自冻结的公开 GitHub issue 标题，检索库只含固定源码文档，
不含 issue 标题、正文或评论。42 条真实查询中仅 9 条具有人工相关文档标签，
标签表示相关使用说明，不代表历史 bug 已解决，也不代表完整智能体成功率。
四种检索方式的原始结果全部保留，不能只报告最好的方式。
性能测量为共享负载下的三轮轮换、单驻留进程、同步完整请求；不是 kernel 吞吐或独占设备结果。

本轮基于 `4cf8f226` 独立分支 `feature/agent-evidence-retrieval`，按顺序合入：

| commit | 范围 |
|---|---|
| `c1107de` | 原子证据存储、权限/环境过滤、词法检索 |
| `c75586e` | 混合召回、上下文预算、可追溯引用 |
| `54f57de` | 原生模型适配、有限精确缓存、更新竞态修复、默认不精排 |
| `4d9654e` | CLI、只读 MCP、真实 stdio 回归 |
| `7adf499` | 真实问题冻结与硬件验证工具 |
| `3b19df7` | 拒绝 MCP 非法版本参数，最终硬件测试代码版本 |

以上 commit 保持独立，建议按依赖顺序 cherry-pick；前两项是后续项的基础，不是可任意打乱的平行补丁。
旧图优化及知识库精排分支保持不变。实测数据和局限见本轮独立验证报告。
