# 安装部署指南

## 目录

- [环境准备](#环境准备)
- [编译本项目](#编译本项目)
- [技术支持](#技术支持)

## 环境准备

目前的软硬件环境支持矩阵

| Host CPU | Device        | OS            |  Support   |
| -------- | ------------  | -----------   | ---------- |
| X86-64   | Nvidia GPU    |  Ubuntu-22.04 |  Yes       |
| X86-64   | Cambricon MLU |  Ubuntu-22.04 |  Yes       |

推荐使用 X86-64 机器以及 Ubuntu-22.04，本文以此环境为例。

1. 确认 GCC 版本为 11.3 及以上的稳定版本，如若您的机器 GCC 版本不满足此条件，请自行编译安装，下述方式二选一：

   - [GCC 官方文档](https://gcc.gnu.org/onlinedocs/gcc-11.3.0/gcc/)

   - [网友安装分享](https://zhuanlan.zhihu.com/p/509695395)

2. 确认 CMake 版本为 3.17 及以上的稳定版本， 如若您的机器 CMake 版本不满足此条件，请自行编译安装，下述方式二选一：

   - [CMake 官方文档](https://cmake.org/install/)

   - [网友安装分享](https://zhuanlan.zhihu.com/p/110793004)

3. 第三方加速卡软件资源安装，目前本项目已经适配了如下的第三方加速卡：

   - 如您的第三方加速卡为英伟达 GPU，请参考英伟达官方文档进行：

     > [驱动安装](https://www.nvidia.cn/geforce/drivers/)，
     > [CUDA Toolkit 安装](https://developer.nvidia.com/cuda-toolkit)，
     > [Cudnn 安装](https://developer.nvidia.com/rdp/cudnn-download)，
     > [Cublas 安装](https://developer.nvidia.com/cublas)，
     > 安装完成后请进行相应的环境变量配置，将可执行文件目录与库目录添加到操作系统识别的路径中，例如
     >
     > ```bash
     > # 将如下内容写入到你的 bashrc 文件并 source 该文件
     > export CUDA_HOME="/PATH/TO/YOUR/CUDA_HOME"
     > export CUDNN_HOME="/PATH/TO/YOUR/CUDNN_HOME"
     > export PATH="${CUDA_HOME}/bin:${PATH}"
     > export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
     > # 如您不方便将上述环境变量配置到 bashrc 文件中进行长期使用，你也可以在我们提供的 env.sh 文件中进行正确配置并激活，作为临时使用
     > source env.sh
     > ```

     我们强烈建议您规范安装，统一到一个目录下，以免不必要的麻烦。

   - 如您的第三方加速卡为寒武纪 MLU，请参考寒武纪官方文档进行：
     > [驱动安装](https://www.cambricon.com/docs/sdk_1.11.0/driver_5.10.6/user_guide_5.10.6/index.html)，
     > [CNToolkit 安装](https://www.cambricon.com/docs/sdk_1.11.0/cntoolkit_3.4.1/cntoolkit_install_3.4.1/index.html)，
     > [CNNL 安装](https://www.cambricon.com/docs/sdk_1.11.0/cambricon_cnnl_1.16.1/user_guide/index.html)，
     > 安装完成后请进行相应的环境变量配置，将可执行文件目录与库目录添加到操作系统识别的路径中，例如
     >
     > ```bash
     > # 将如下内容写入到你的 bashrc 文件并 source 该文件
     > export NEUWARE_HOME="/usr/local/neuware"
     > export PATH="${NEUWARE_HOME}/bin:${PATH}"
     > export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH}"
     > # 如您不方便将上述环境变量配置到 bashrc 文件中进行长期使用，你也可以在我们提供的 env.sh 文件中进行正确配置并激活，作为临时使用
     > source env.sh
     > ```

     我们强烈建议您规范安装，统一到一个目录下，以免不必要的麻烦。另外请注意，由于 MLU 上层软件建设适配程度有限，如您在其覆盖的机器，操作系统之外运行，需要在安装驱动之后使用上层软件的 Docker。

4. 确认您安装了 make，build-essential， python-is-python3， python-dev-is-python3， python3-pip， libdw-dev，如您的机器没有上述基础依赖，请自行按需安装。

   - 在使用 apt-get 工具情况下，您可以这样执行

     ```bash
     sudo apt-get install make cmake build-essential python-is-python3 python-dev-is-python3 python3-pip libdw-dev
     ```

5. 更新pip并切换到清华源

   ```bash
   python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

6. 安装一些不必要的项目（可选）

   - 如您需要运行本项目下的 example 代码，您需要安装一些辅助项目。请注意这些项目不是必要的，若您不需要运行样例代码，这些项目无需安装。

     > [Pytorch](https://pytorch.org/get-started/locally/)：业界内流行的神经网络编程框架
     > [ONNX](https://onnx.ai/get-started.html)：业界内流行的神经网络模型存储文件与转换器
     > [onnxsim](https://pypi.org/project/onnxsim/)：一个简化onnx模型的小工具
     > [onnx2torch](https://github.com/ENOT-AutoDL/onnx2torch)：一个将onnx模型转换pytorch模型的小工具
     > [tqdm](https://pypi.org/project/tqdm/)：一个显示程序运行进度条的小工具

   - 如您需要使用本项目下的 InfiniTest 测试工具，你还需要安装如下的项目：

     > [protobuf](https://github.com/protocolbuffers/protobuf)： 一种序列化文件的格式及其编译、序列化、解析工具

## 编译本项目

推荐使用 X86-64 机器以及 Ubuntu-22.04，本文以此环境为例。

1. 配置环境

   打开 env.sh 文件进行环境变量配置，之后执行

   ```bash
   source env.sh
   ```

2. 编译本项目并打包成 Python 库进行安装

   我们提供了意见编译参数，您可以在项目根目录下执行下面的命令。第一次执行会同时安装 python 依赖库，耗时略长，请耐心等待。

   仅编译 CPU 部分，不编译第三方计算卡：

   ```bash
   make install-python
   ```

   编译 CPU 部分，同时编译英伟达 GPU 部分：

   ```bash
   export CUDA_HOME=/path/to/your/cuda_home
   make install-python CUDA=ON
   ```

   编译 CPU 部分，同时编译寒武纪 MLU 部分：

   ```bash
   export NEUWARE_HOME=/path/to/your/neuware_home
   make install-python BANG=ON
   ```

3. 使用方法

   安装成功后，您就可以使用本项目的 Python 接口进行编码并运行。具体使用方式可以参考项目样例代码 example/Resnet/resnet.py 以及用户使用手册

## Docker

本项目也提供了 Docker 的环境，您可以使用 `make docker-build` 或 `make docker-build CUDA=ON` 命令启动并编译 Dockerfile，您可以通过添加编译选项或者修改 Makefile 变量修改 docker image 名称或者所选的 Dockerfile 文件。
 
由于在拉取 github repo 时需要将 ssh key 加入到 github profile 中，因此暂时注释掉拉取 repo 并编译项目的过程，由用户在进入 docker 后自己维护 ssh key（将 host 中的 ssh key 复制到 docker 中可能会遇到环境不一致的问题）。

```shell
# Build docker container.
make docker-build
# Run docker image.
make docker-run
# Execute docker image.
make docker-exec
```

如果需要编译 CUDA 版，请使用如下命令：
```shell
# Build docker container.
make docker-build CUDA=ON
# Run docker image.
make docker-run CUDA=ON
```

## 技术支持

如遇到问题，请联系我们技术支持团队
