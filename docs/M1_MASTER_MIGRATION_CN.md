# M1 迁移到 master

本次迁移的 InfiniTensor 基线是 `703596a91fc20288d0cb17bd1993b011721ef378`。
InfiniOps 同步使用 master 的锁定版本 `f43ba8c1174c1cce2a89b690932fe59787bc9361`，
InfiniRT 使用 `0cdbb16967e15f2e055dea1ec9641617bf3b6cf6`。
完整版本与 InfiniOps 补丁摘要见 `dependencies/m1-lock.json`。

## 迁移内容

保留动态 ONNX 导入、CPU shape 元数据求值、单图 prepare、权重驻留、失败回滚、
Encoder bridge、能力诊断、EmbeddingSession、离线 JSONL 编码和查询、模型导出及验收工具。
InfiniOps 补丁重新应用到新版依赖，保留 master 的 CMake、真实 NPU stream 适配与
NPU tensor from_blob offset 修复。其他设备保留 M1 的明确 target_device 和设备延迟初始化。
InfiniTensor 保留 master 的设备复制、workspace 与 graph capture 修复。
重新验证还修复了 InfiniOps 原生 MLU RMSNorm 在调用队列中无条件同步的问题：
元数据复制和 kernel 按队列顺序提交，由调用者负责完成同步，避免同步使捕获失效。

NPU 使用 torch_npu `2.7.1.post10`，进程导入前设置 `TASK_QUEUE_ENABLE=0`。
使用安装好的 CANN 环境及匹配的 ABI，不能只替换 Python 包而复用旧二进制。
CUDA、MLU、NPU 的正式文本接口均应用 FP32 策略并恢复调用前设置。

## 重新验证

六端点分别使用独立的 clean 构建目录和安装前缀，不复用旧 InfiniTensor、InfiniOps、
InfiniRT 二进制。模型和检索数据采用同一份已冻结包；L20 另执行一次本地重新导出与精度验证。
设备测试不停止其他已有任务，性能结论仅适用于记录下来的共享负载环境。

验收入口为 `scripts/accept_m1.py`，包含 preflight、accuracy、retrieval、stability、
benchmark、install、all。动态精度默认执行 142 组输入，包括演示、边界、35 组矩阵和
100 组固定种子随机形状。检索采用固定的 100 条查询和 1,000 条文档。

长稳 method_version=2 使用一个引擎实例，预热后同时满足连续两小时和一万次请求。
请求内容持续按种子 315 变化，每 100 次与同设备 PyTorch 比较，检查旧输出和权重地址。
资源在同一 `(8,128)` 形状下每约五秒采样，并保存完整 JSONL；首末 30 分钟中位数
和最后 60 分钟线性趋势参与判断。短程校准只能标记 CALIBRATED。
厂商分配器、InfiniRT 分配计数和设备进程内存分开报告。MLU 在验收专用目录提供
宿主机同版本的 cnmon 工具，使用 `cnmon all` 中容器命名空间 PID 的本进程显存数据。
三类设备均要求直接的进程显存证据，不使用整卡显存代替。

性能矩阵按引擎轮换顺序执行三轮，各轮每形状新建进程、预热 20 次、测量 200 次。
混合请求使用固定种子生成不同最长长度，记录每次实际输入形状与有效/填充 token 数。
ORT CUDA 显式关闭 TF32，使用设备 I/O binding；provider profiling 单独执行且保留
输入 OrtValue 生命周期。计时包含 host 提交与同步边界，不解释为纯 kernel 时间。

安装检查还执行 master C++ 和 Python copy/workspace graph 回归、InfiniRT CTest、
M1 Python 测试及 GPU 算子验证。认证驱动支持 capture，三个后端均强制执行对应回归，
不能通过能力 skip 隐藏失败；M1 精度、检索和长稳不得跳过或回退到 CPU 前向。

执行源码与完整交付源码分别记录 SHA256；所有正式阶段必须匹配最终快照。
最终报告由 `scripts/finalize_m1_acceptance.py` 聚合，缺少阶段、资源、回归或版本证据
会返回 FAIL。实际硬件名称以设备查询为准，不能仅根据服务器约定名称宣称型号。

## 复现

模型准备和日常接口见 `BGE_EMBEDDING_M1_CN.md`。在匹配厂商环境中执行：

```bash
bash scripts/install_m1_clean.sh --source-dir "$PWD" --work-dir /absolute/new-clean \
  --python /absolute/venv/bin/python --device nvidia --device-id 0 \
  --model-dir /absolute/model --build-only
# NVIDIA 预先设置 CUDA_HOME、CUDA_ARCH，并准备锁定的 CUTLASS archive。
# SDK 和 torch 动态库路径由实际环境提供，不能携带旧项目库路径。
export PYTHONPATH=/absolute/new-clean/source/pyinfinitensor/src
export LD_LIBRARY_PATH=/absolute/new-clean/prefix/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python /absolute/new-clean/source/scripts/accept_m1.py --stage accuracy \
  --device nvidia --device-id 0 --model-dir /absolute/model --output /absolute/results
python /absolute/new-clean/source/scripts/accept_m1.py --stage stability \
  --device nvidia --device-id 0 --model-dir /absolute/model --output /absolute/results \
  --stability-requests 10000 --stability-seconds 7200 --run-oom-probe
python /absolute/new-clean/source/scripts/benchmark_m1_matrix.py \
  --model-dir /absolute/model --device nvidia --device-id 0 --output /absolute/results
```

六端点的完整环境、构建命令、冻结包与原始结果一同交付，不能把旧 release 的通过记录
直接当成 master 迁移的通过记录。旧源码与旧证据保留用于追溯。
