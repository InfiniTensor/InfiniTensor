# 开发指南

## 组件职责

| 组件 | 负责内容 |
| --- | --- |
| InfiniTensor | ONNX 与图接口、形状和属性语义、图变换、执行计划及资源生命周期 |
| InfiniOps | 算子公开接口、实现选择、dtype／布局约束、数值计算及捕获安全声明 |
| InfiniRT | 设备、内存、流、同步和设备图的捕获与重放 |

厂商 SDK 和算子 kernel 放在下层组件。InfiniTensor 使用组件公开接口，前端与统一执行器不承担逐平台实现选择。

## 接入算子与模型

InfiniTensor 从已安装 InfiniOps SDK 的公开工厂声明生成绑定和 schema。输出文件位于构建目录的 `generated/infiniops_bindings.cc` 与 `generated/infiniops_bindings.json`；清单记录已支持签名及不支持的参数类型。

扩展时按职责选择入口：

1. **直接构图**：算子已有公开工厂且参数类型受生成器支持时，使用 `InfiniOp`／`infini_op()`，由前端提供输出 shape 和 dtype。
2. **导入 ONNX**：补充节点属性、输入解释和形状推导；算子签名存在并不等于 ONNX 语义已经接入。
3. **兼容已有图节点**：节点语义与 SDK 接口有差异时，在 `src/kernels/infiniops/lowering.cc` 做参数、广播、布局或组合映射。
4. **补计算能力**：缺少算子实现、dtype 支持或硬件能力时，在 InfiniOps 完成；设备和流接口缺失则在 InfiniRT 完成。

修改生成器应覆盖新的参数类型或声明格式，避免加入只识别某个算子名字的特殊分支。

## 执行与资源

执行器按图准备算子实例和张量视图。临时张量及外部工作区按生命周期复用，常量独立保留；provider 自行管理的内部内存由其实现负责。

普通执行和捕获使用同一准备路径。捕获保留实例、张量存储和工作区，重放期间地址需有效；拓扑、shape、存储或执行线程变化时，应重新准备相应状态。异常处理也必须完成已提交任务所需的同步，再允许释放资源。

ATen provider 的流互操作由下层组件负责。C++ 宿主使用依赖 Python 的 provider 时，需初始化匹配环境的解释器并导入 provider 模块；需要 Python API 的调用线程应能获取 GIL。C++ 测试共用 `include/test_infini_aten.h` 完成环境初始化。

## 构建与测试

```bash
make build TEST=ON
ctest --test-dir build/Release --output-on-failure
make test-api
make test-onnx
```

按后端配置 `INFINI`、`ATEN` 和组件前缀。指定其他 `TYPE` 或构建目录时，调整 `ctest --test-dir`。完整 ONNX 测试还需要测试模型，获取方式见 [CPU CI](../.github/workflows/build.yml)。

测试按以下边界组织：

| 目录 | 验证内容 |
| --- | --- |
| `test/core` | 图、张量、分配器、图变换等核心行为 |
| `test/operators` | InfiniTensor 节点的属性和形状语义 |
| `test/kernels/nativecpu` | 仓库内置 CPU kernel |
| `test/infini` | 绑定、执行计划、图缓存、跨节点顺序和资源生命周期 |
| `pyinfinitensor/tests` | Python API、ONNX 导入导出、绑定生成器和设备集成 |
| `test/nnet` | 张量程序优化研究模块，通过相应构建选项启用 |

算子的数值矩阵、dtype 组合及 provider 参数约束由 InfiniOps 测试负责。InfiniTensor 仅选择能体现接口映射、图语义或执行集成的代表用例；同一契约不在 C++／Python 层重复铺设算子矩阵。

运行绑定及执行集成测试：

```bash
python pyinfinitensor/tests/test_generate_infiniops_bindings.py
python pyinfinitensor/tests/test_infiniops_bindings.py
python pyinfinitensor/tests/test_infiniops_stream.py
```

生成器测试不需要设备；另外两组需按用例准备相应 SDK 实现。stream 测试用于加速设备上的 ATen SDPA 与 InfiniRT copy 顺序，缺少所需设备或实现时会失败。Graph 测试按运行时能力执行，结果中需区分实际执行与能力不适用。

## CI 维护

- `build.yml`：内置 CPU、Python 扩展安装和 ONNX 前端构建与测试。
- `infiniops-native.yml`：重新构建 RT／Ops 和 InfiniTensor，检查无 ATen 的集成。
- `infiniops-aten-cuda.yml`：在 CUDA runner 上重新构建组件并执行 ATen stream 测试。
- `clang-format-check.yml`：C++ 格式检查，使用 clang-format 14。

CUDA 任务需要带 `self-hosted`、`linux`、`x64`、`cuda` 标签的 runner，并设置仓库变量 `INFINITENSOR_ATEN_CUDA_CI=true`。runner 提供 CMake 3.24+、C++ 编译器、CUDA toolkit（`nvcc` 在 PATH 中）、驱动及 CUDA PyTorch；`INFINITENSOR_CUDA_PYTHON` 可指定已有 PyTorch 的 Python 3.10–3.12 解释器。任务处理同仓库 PR、master 推送和手工触发，构建日志与环境清单作为 artifact 保存。

## 依赖更新与提交

InfiniOps 和 InfiniRT 由各自子模块引用记录。先完成下层组件修改，再在 InfiniTensor 更新对应引用并运行受影响的集成测试。试验组件时，可通过独立安装前缀验证，不修改子模块引用。

提交说明写清能力或行为变化、职责边界和实际验证范围。硬件环境、组件摘要、性能原始数据及阶段性审核结果保留在测试记录或 PR 中，使用文档只描述可用接口、配置及限制。
