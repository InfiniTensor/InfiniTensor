# M1：动态中文 Embedding 推理

本目录的交付目标是以一份 ONNX 图和一份驻留权重，在 NVIDIA、寒武纪 MLU 和昇腾
单卡上持续执行 `BAAI/bge-small-zh-v1.5`。模型输出为 CLS 向量，Python 接口在 CPU
完成 L2 归一化，返回 `float32 [N, 512]`。

## 准备模型

在一台可访问 Hugging Face 的受控机器上，先记录不可变 commit SHA，然后导出：

```bash
python scripts/export_bge_small_zh.py \
  --revision <huggingface-commit-sha> --output-dir artifacts/bge-small-zh \
  --validate
```

验收机断网时可从已经核验摘要的冻结包重走导出：增加
`--source /absolute/frozen-model-package`，并设置 `HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1`。输出仍记录显式 revision，并重新执行 ONNX checker 和
PyTorch→ORT CPU 数值对照。

该目录包含 tokenizer、配置、`model.onnx` 和带 SHA256 的 `manifest.json`。复制整个
目录到离线推理机即可；推理阶段使用 `local_files_only=True`，不需要联网。

## 安装与自检

本轮迁移基于 2026-10-12 获取的 master `703596a91fc20288d0cb17bd1993b011721ef378`。
依赖以 `dependencies/m1-lock.json` 为准。InfiniRT 保持 master 的版本；InfiniOps
在 master 指定的 `f43ba8c` 上应用 `dependencies/InfiniOps-m1.patch`，包含 Gather/Cast
和厂商 ATen 适配。仅使用未打补丁的 PR 原始依赖不能运行本模型。先核对补丁
SHA256，再在 InfiniOps 源码目录执行 `git apply /absolute/path/InfiniOps-m1.patch`，
重新构建并安装 InfiniOps，随后构建本项目。离线验收包还包含固定 revision 的 Git
bundles；NVIDIA 构建额外包含锁文件所列 CUTLASS commit 的源码归档及 SHA256。

在配置好匹配的 InfiniOps 安装前缀和 Python provider 后：

```bash
make build INFINI=ON ATEN=ON
make install-python PROVIDER_MODULES=torch CHECK_BACKEND=nvidia CHECK_DEVICE=0
python scripts/check_install.py --provider-modules torch --require-infini --device 0 --json
```

自检失败会在加载模型前报告 Python provider、后端和设备创建阶段。它不能替代模型级
能力验证。

从冻结源码重新构建时使用独立工作目录，脚本拒绝覆盖已有目录：

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=89 \
  bash scripts/install_m1_clean.sh --source-dir "$PWD" \
  --work-dir /tmp/m1-clean-l20 --python /path/to/python \
  --device nvidia --device-id 0 --model-dir /absolute/model
```

MLU/昇腾分别将 `--device` 改为 `cambricon`/`ascend`，并先加载厂商 SDK。脚本从锁文件
检出 InfiniRT/InfiniOps，校验并应用补丁，在新前缀构建后运行安装自检、离线模型执行和
回归测试。`install.json` 同时记录实际加载的 Python 模块和本地动态库路径，防止误用旧产物。
若设备正在执行长稳测试，可先增加 `--build-only` 完成 CPU 编译；长稳结束后用
`scripts/verify_m1_clean.sh` 对该工作目录执行设备自检、CTest 和仓库外离线编码/查询。

## 编码与检索

```bash
python examples/python/bge_embedding.py --model-dir artifacts/bge-small-zh \
  --query "如何进行向量检索" \
  --document "向量数据库支持语义搜索" \
  --document "今天天气很好"
```

`EmbeddingSession.encode(texts, kind="passage", batch_size=32)` 按实际最长序列右侧
padding，不使用静态分桶；`kind="query"` 自动加上 BGE 官方中文检索指令。M1 的认证
边界为 `1 <= B <= 32`、`1 <= S <= 512`。长文本会连同 special token 与查询前缀一起截断。

离线文档库使用含字符串 `id`/`text` 的 JSONL：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/python/encode_jsonl.py --model-dir artifacts/bge-small-zh \
  --input documents.jsonl --output documents.npz --device nvidia
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python examples/python/query_jsonl.py --model-dir artifacts/bge-small-zh \
  --vectors documents.npz --query "如何进行向量检索" --top-k 10 --device nvidia
```

向量产物保存文档 ID、模型摘要和 ID 摘要；查询结果按分数降序稳定排列。

## 验收与性能记录

统一入口的阶段为 `preflight、accuracy、retrieval、stability、benchmark、install、all`。
每阶段都生成 JSON，必需证据缺失或门槛失败时返回非零：

```bash
python scripts/accept_m1.py --stage accuracy --device nvidia --device-id 0 \
  --model-dir artifacts/bge-small-zh --output reports/l20
python scripts/accept_m1.py --stage retrieval --device nvidia --device-id 0 \
  --model-dir artifacts/bge-small-zh --dataset-dir artifacts/T2Retrieval \
  --output reports/l20
python scripts/accept_m1.py --stage stability --device nvidia --device-id 0 \
  --model-dir artifacts/bge-small-zh --output reports/l20 --run-oom-probe
```

`stability` 同时要求至少 10,000 次成功请求和至少两小时连续运行。T2Retrieval 固定 revision
及选集规则写入结果，比较同设备 PyTorch 的向量和 Top-10。NVIDIA 的性能矩阵包含
InfiniTensor、PyTorch CUDA、ORT CUDA；MLU/NPU 包含 InfiniTensor 与同设备 PyTorch：

```bash
python scripts/benchmark_m1_matrix.py --model-dir artifacts/bge-small-zh \
  --device nvidia --device-id 0 --output reports/l20
```

每个引擎/形状/轮次使用新进程，三轮轮换引擎顺序；每轮预热 20 次、测量 200 次。
报告保留原始样本、外部负载、tokenizer、H2D、执行、D2H、归一化、进程内存和设备内存。
ORT 正式样本使用设备 I/O binding，另起 session 做 provider profiling，encoder 节点必须
实际分配到 CUDA。共享设备上的数据只作为共享负载基线，M1 不以快于对照引擎为门槛。

汇总六端点时运行：

```bash
python scripts/finalize_m1_acceptance.py --source . \
  --results-root validation/results/final --model-dir artifacts/bge-small-zh \
  --dataset-dir artifacts/T2Retrieval --output-json reports/m1-final.json \
  --output-md reports/m1-final.md
```

聚合器验证锁文件与 InfiniOps 补丁摘要，计算交付源码树摘要；六端点的预检、精度、检索、
长稳、OOM、性能和安装阶段全部通过时才输出总体 PASS。

## 动态模型约束

`OnnxStub.prepare(input_shapes)` 只在 CPU 执行由输入维度和常量产生的 shape 子图。
它支持目标导出图所需的 Shape、Gather、Slice、Concat、Unsqueeze/Squeeze、Cast、基本
整数算术/比较、Where、Range、ConstantOfShape、Expand 与 Reshape。shape 参数依赖
GPU activation 的图会在准备阶段按节点名报错；不会回读 activation 模拟执行。

动态 `Reshape`、`Expand`、`Slice` 会原地更新图内参数，再形状推导和重新分配激活。首次
初始化后 initializer 不再 `copyin`；返回的 NumPy 数组拥有自己的内存，因此不会被下一
请求覆盖。未 prepare 的动态图直接调用底层 `run()` 会明确报错。

最终发布状态以自动生成的 `m1-final.json` 为准，不能用早期精度报告替代缺失阶段。
量化、并发调度、HTTP 服务、Graph Capture 优化和 reranker 不在 M1 范围内。

MLU 使用 `INFINIOPS_PROVIDER_MODULES=torch_mlu` 并向示例传入 `--device cambricon`；
昇腾使用 `INFINIOPS_PROVIDER_MODULES=torch_npu` 和 `--device ascend`，先加载匹配的
CANN 环境。昇腾采用 master 的外部流接口，需要匹配的 `torch_npu 2.7.1.post10`
并在导入 provider 之前设置 `TASK_QUEUE_ENABLE=0`。本轮迁移保留 master 的图捕获
和缓存内存修复；BGE 验收仍使用普通前向路径。所有 encoder 节点在设备上执行，CPU 只负责 shape 元数据、
tokenizer 和最终归一化。
