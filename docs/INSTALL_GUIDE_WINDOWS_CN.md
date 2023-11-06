# Windows系统安装部署指南

## 目录

- [环境准备](#环境准备)
- [编译本项目](#编译本项目)
- [技术支持](#技术支持)

## 环境准备

1. 确认 Visual Stuio 2019 版本为 16.10 及以上的稳定版本，安装时选择“使用 C++ 的桌面开发”。

   - [Visual Studio 下载](https://visualstudio.microsoft.com/zh-hans/downloads/)

2. 确认 CMake 版本为 3.17 及以上的稳定版本， 如若您的机器 CMake 版本不满足此条件，请自行编译安装，下述方式二选一：

   - [CMake 官方文档](https://cmake.org/install/)

   - [网友安装分享](https://zhuanlan.zhihu.com/p/656121868)

3. 第三方加速卡软件资源安装，目前本项目已经适配了如下的第三方加速卡：

   - 如您的第三方加速卡为英伟达 GPU，请参考英伟达官方文档进行：

     > [驱动安装](https://www.nvidia.cn/geforce/drivers/)，
     > [CUDA Toolkit 安装](https://developer.nvidia.com/cuda-toolkit)，
     > [Cudnn 安装](https://developer.nvidia.com/rdp/cudnn-download)，
     > [Cublas 安装](https://developer.nvidia.com/cublas)，

     > CUDA Toolkit 、Cudnn 和 Cublas 一般安装在相同目录。CUDA Toolkit 安装完成后，会自动将可执行文件目录与库目录添加到系统环境变量 `PATH` 中。例如，CUDA Toolkit 安装在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\` ，您查看系统变量 `PATH` 中会看到如下内容。
     >
     > ```
     > C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
     > ```

     我们强烈建议您规范安装，统一到一个目录下，以免不必要的麻烦。


4. 确认您安装了 Python（64位），并将 python 的安装目录添加到系统环境变量 `PATH` 中。

   - [Python 官方文档](https://www.python.org/downloads/)


5. 更新pip并切换到清华源。

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


1. 打开命令提示符（CMD），运行工具集脚本设置环境。例如，Visual Stuio 安装在 `C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional`，可执行以下命令：

   ```bash
   C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat
   ```

   工具集相关内容可参考文档[通过命令行使用 Microsoft C++ 工具集](https://learn.microsoft.com/zh-cn/cpp/build/building-on-the-command-line?view=msvc-170)。

2. 编译本项目并打包成 Python 库进行安装

   我们提供了意见编译参数，您可以在项目根目录下执行下面的命令。第一次执行会同时安装 python 依赖库，耗时略长，请耐心等待。

   新建 build 目录：

   ```bash
   mkdir build
   cd build
   ```

   仅编译 CPU 部分，不编译第三方计算卡：

   ```bash
   cmake .. -G "NMake Makefiles" -DUSE_BACKTRACE=OFF -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release
   nmake
   copy backend*.pyd ..\pyinfinitensor\src\pyinfinitensor\
   pip install -e ..\pyinfinitensor
   ```

   编译 CPU 部分，同时编译英伟达 GPU 部分：

   ```bash
   cmake .. -G "NMake Makefiles" -DUSE_CUDA=ON -DUSE_BACKTRACE=OFF -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release
   nmake
   copy backend*.pyd ..\pyinfinitensor\src\pyinfinitensor\
   pip install -e ..\pyinfinitensor
   ```

3. 使用方法

   安装成功后，您就可以使用本项目的 Python 接口进行编码并运行。具体使用方式可以参考项目样例代码 example/Resnet/resnet.py 以及用户使用手册


## 技术支持

如遇到问题，请联系我们技术支持团队
