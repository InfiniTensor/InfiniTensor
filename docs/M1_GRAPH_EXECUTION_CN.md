# M1 动态准备优化与统一设备图执行

EmbeddingSession 和 OnnxStub 默认使用 `normal`，保留旧调用行为。显式选择 `auto` 时，连续四次相同输入形状后捕获，此后重放；形状变化时先普通执行并重新累计次数。普通执行支持原有全部合法动态 B/S，不使用静态桶，也不保留多份模型。

```python
session = EmbeddingSession(model_dir, device="cambricon", execution_mode="auto")
vectors = session.encode(["中文文档"])
print(session.execution_stats())
session.stub.set_execution_mode("normal")
```

三种模式适用于 NVIDIA、昇腾和寒武纪：

| 模式 | 行为 |
|---|---|
| normal | 普通 InfiniTensor → InfiniOps 设备前向 |
| auto | 同一具体形状连续成功执行达到 `graph_capture_after` 后捕获；变化时重置 |
| graph | 对每个当前具体状态立即调用统一设备图接口；频繁变形可能更慢 |

捕获失败直接抛出原错误，并记录 `graph_failures`；没有隐式 CPU、ORT、PyTorch 前向回退。可以显式切到 normal 后再次执行。非法输入和未准备图在执行前拒绝。一次会话保留一份捕获实例；清理或形状变化只清理本图，不清理共享 runtime 上的其他图。

## 调用路径

`EmbeddingSession → OnnxStub → GraphHandler.run_with_graph → InfiniRuntimeObj::runWithGraph → infini::rt::runtime`

捕获、实例化、重放和释放均调用 InfiniRT 的 `StreamBeginCapture`、`StreamEndCapture`、`GraphInstantiate`、`GraphLaunch`、`GraphExecDestroy`、`GraphDestroy`。InfiniTensor 不直接调用 CUDA/ACL/CNRT 图 API。

锁定 InfiniRT `0cdbb16967e15f2e055dea1ec9641617bf3b6cf6` 已提供：

| 后端 | InfiniRT 内部实现 |
|---|---|
| NVIDIA | CUDA Stream Capture / CUDA Graph |
| 寒武纪 | CNRT Queue Capture / TaskTopo Entity |
| 昇腾 | ACL Capture / Model RI ExecuteAsync，按实际 SDK 动态解析 |

统一接口不代表所有 SDK/provider 组合天然可捕获。实际认证需要完整模型数值、输入更新、形状切换、生命周期和资源验证。

## 资源与精度

LayerNorm mean/rstd 使用 runtime 的 capture workspace：warmup 分配、capture 复用、图实例持有至释放。ATen provider 的捕获内存仍由 InfiniOps 管理；前端不依赖 torch.cuda.CUDAGraph、torch_npu.NPUGraph 或厂商 Python 图接口。

昇腾上，torch_npu 的 `releasePool` 只把旧捕获池标记为可回收；捕获期间不能再通过 OOM 回收它。因此 InfiniOps 在旧图句柄销毁后调用 allocator 的 `emptyCache(true)`，回收已空闲的私有池，避免连续换形耗尽显存。该接口会同步并清理本进程所有已用 NPU 的空闲缓存，范围不限于当前模型；仍存活的 tensor 和其他图持有的捕获池不会释放。图句柄按模型隔离，但空闲 allocator 缓存并非按模型隔离。换形性能计入这项成本；M1 不承诺并发捕获。

昇腾外部流适配还会把 torch_npu 的线程局部设备缓存同步到 InfiniRT 已激活的设备。原生 `aclrtSetDevice` 不更新该缓存；不对齐时，非零设备上的 NPUStreamGuard 可能恢复设备 0，使后续原生流拷贝失败。同步仅位于 InfiniOps provider 适配层，图捕获和重放仍使用 InfiniRT。

FP32 策略在捕获和重放时均生效，退出调用后恢复应用的原设置。权重单次上传，NumPy 输出独立拥有内存。

`prepare()` 的验证与绑定共用一次 shape 求值结果；静态依赖集、常量、属性在加载时建立。失败回滚会对旧形状重新求值，不复用失败准备的临时结果。

## 验证入口

```bash
python scripts/accept_m1.py --stage accuracy --device ascend --model-dir MODEL \
  --execution-mode graph --output results/graph-accuracy
python scripts/benchmark_m1_matrix.py --device cambricon --model-dir MODEL \
  --execution-mode auto --output results/auto-benchmark
INFINITENSOR_REQUIRE_GRAPH_CAPTURE=1 python -m unittest discover \
  -s pyinfinitensor/tests -p 'test_workspace_graph_capture.py'
```

显式 graph 的混合形状数据包含重新捕获成本；固定形状热请求与首次捕获分开报告。新增路径的逐端点实际结果与源码摘要保存在 `validation/optimization-20261016`，未通过的组合不能标为已认证。
