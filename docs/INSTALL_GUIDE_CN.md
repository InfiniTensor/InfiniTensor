# 安装指南

## 环境与后端

InfiniTensor 支持两种执行方式：内置 CPU runtime，以及由 InfiniOps／InfiniRT 提供的执行后端。加速设备的驱动、SDK 和 provider 在下层组件中安装与配置。

构建 InfiniTensor 需要 CMake 3.17+、支持 C++17 的编译器、Python 3.9+（含开发头文件）、pip 和 make。下层组件可以有更高的工具链要求；使用 ATen provider 时，Python、PyTorch、厂商扩展和 C++11 ABI 必须匹配。普通 Release 构建无需栈回溯依赖，启用 `BACKTRACE=ON` 时需安装 `libdw-dev`。

## 获取源码

```bash
git clone https://github.com/InfiniTensor/InfiniTensor.git
cd InfiniTensor
git submodule update --init --recursive
```

子模块记录与项目配套的 InfiniOps 和 InfiniRT 源码版本。组件分别构建、安装，再由 InfiniTensor 链接；具体步骤见 [InfiniRT](../3rd-party/InfiniRT/README.md) 和 [InfiniOps](../3rd-party/InfiniOps/README.md)。

## 安装内置 CPU runtime

不使用 InfiniOps 时：

```bash
make install-python INFINI=OFF PYTHON="$(command -v python3)"
```

该命令构建扩展、安装 Python 包并执行基本安装检查。

## 安装 InfiniOps 后端

先构建 InfiniRT，再构建使用该 runtime 的 InfiniOps。下面以原生 CPU provider 为例；先按下层组件文档准备其构建依赖：

```bash
export INFINI_PREFIX="$HOME/.infini"
cmake -S 3rd-party/InfiniRT -B build/InfiniRT \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON \
  -DCMAKE_INSTALL_PREFIX="$INFINI_PREFIX"
cmake --build build/InfiniRT --parallel
cmake --install build/InfiniRT

cmake -S 3rd-party/InfiniOps -B build/InfiniOps \
  -DCMAKE_BUILD_TYPE=Release -DWITH_CPU=ON -DWITH_TORCH=OFF \
  -DGENERATE_PYTHON_BINDINGS=OFF -DINFINI_RT_ROOT="$INFINI_PREFIX" \
  -DCMAKE_INSTALL_PREFIX="$INFINI_PREFIX"
cmake --build build/InfiniOps --parallel
cmake --install build/InfiniOps

make install-python INFINI=ON ATEN=OFF \
  INFINIOPS_ROOT="$INFINI_PREFIX" INFINIRT_ROOT="$INFINI_PREFIX" \
  CHECK_BACKEND=infini PYTHON="$(command -v python3)"
```

加速设备同样按此顺序安装，设备选项、架构参数及算子选择由组件的构建配置决定。InfiniTensor 从 InfiniOps SDK 的公开头文件生成绑定，需在构建 Python 环境中安装 `libclang` 和 `PyYAML`。

使用已经安装好的 ATen provider：

```bash
make install-python INFINI=ON ATEN=ON \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix \
  CHECK_BACKEND=infini PYTHON="$(command -v python3)"
```

两个安装前缀可以相同，也可以分别指定。它们必须来自兼容的构建；ATen provider 还需与所选 Python/PyTorch 使用相同的 C++11 ABI。构建时会自动读取 PyTorch ABI，也可通过 `INFINIOPS_CXX11_ABI` 显式指定。

## Python provider 配置

依赖厂商 Python 扩展的 provider 需要在加载后端前导入对应模块。安装时使用 `PROVIDER_MODULES`，运行时使用 `INFINIOPS_PROVIDER_MODULES`：

```bash
export INFINIOPS_PROVIDER_MODULES=torch_mlu
make install-python INFINI=ON ATEN=ON \
  INFINIOPS_ROOT=/path/to/infiniops-prefix \
  INFINIRT_ROOT=/path/to/infinirt-prefix \
  PROVIDER_MODULES="$INFINIOPS_PROVIDER_MODULES" CHECK_BACKEND=infini \
  PYTHON="$(command -v python3)"
```

寒武纪 ATen provider 使用 `torch_mlu`，昇腾使用 `torch_npu`。昇腾外部 runtime stream 要求在启动 Python 前设置 `TASK_QUEUE_ENABLE=0`。纯原生 provider 无需这些 Python 扩展。

provider 的共享库必须能被动态链接器找到；库目录不在所选 Python 环境中时，通过 `INFINIOPS_PROVIDER_LIBRARY_DIRS` 传入以分号分隔的路径。保持构建、安装检查和应用运行使用同一环境。

## 验证安装

```bash
# 内置 CPU runtime
python scripts/check_install.py --backend cpu

# InfiniOps SDK 默认设备
python scripts/check_install.py --backend infini --device 0
```

也可将 `--backend` 设为 InfiniRT 提供的具体设备名称。脚本加载已安装的扩展并执行矩阵乘法；导入、设备创建和执行失败都会返回非零。检查通过后，按[使用指南](USER_GUIDE_CN.md)运行应用模型。

## 常用构建参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `PYTHON` | `python3` | 构建、安装与自检使用的解释器 |
| `INFINI` | `AUTO` | 找到安装前缀时启用 InfiniOps；显式 `ON` 要求依赖完整 |
| `ATEN` | `ON` | 链接 InfiniOps 的 ATen provider 依赖 |
| `TYPE` | `Release` | 构建类型 |
| `TEST` | `ON` | 构建测试可执行文件 |
| `JOBS` | `8` | 编译并行度 |
| `BACKTRACE` | `OFF` | 启用依赖 libdw 的栈回溯 |
| `CHECK_BACKEND` | 空（自检使用 `cpu`） | 安装自检的执行后端 |

以上为 Makefile 入口的默认值；直接调用 CMake 时应显式设置所需选项。`make clean` 清理整个 `build/` 目录。

## 常见问题

| 现象 | 检查事项 |
| --- | --- |
| 找不到 InfiniOps／InfiniRT | 安装前缀、头文件、共享库和 CMake 包是否完整 |
| 扩展导入失败或出现未定义符号 | Python、PyTorch、C++11 ABI 与动态库搜索路径是否匹配 |
| provider 加载失败 | 厂商扩展是否能独立导入，运行环境是否与构建环境一致 |
| 设备不可见 | 设备名称、卡号、驱动、SDK 及运行权限 |
| 算子创建或执行失败 | 所选 provider 是否支持该算子、dtype、形状和外部流 |

离线部署时，预先准备源码子模块、构建依赖、Python 包和模型文件，再将已安装组件及其运行库一起部署。`make install-python` 使用可编辑安装，运行目录需保留对应源码和扩展文件。
