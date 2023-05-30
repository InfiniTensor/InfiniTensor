﻿# 安装部署手册

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

   > [GCC 官方文档](https://gcc.gnu.org/onlinedocs/gcc-11.3.0/gcc/)

   > [网友安装分享](https://zhuanlan.zhihu.com/p/509695395)

2. 确认 CMake 版本为 3.17 及以上的稳定版本， 如若您的机器 CMake 版本不满足此条件，请自行编译安装，下述方式二选一：

   > [CMake 官方文档](https://cmake.org/install/)

   > [网友安装分享](https://zhuanlan.zhihu.com/p/110793004)

3. 第三方加速卡软件资源安装，目前本项目已经适配了如下的第三方加速卡：

   > 如您的第三方加速卡为英伟达 GPU，请参考英伟达官方文档进行[驱动安装](https://www.nvidia.cn/geforce/drivers/)，[CUDA Toolkit 安装](https://developer.nvidia.com/cuda-toolkit)，[Cudnn 安装](https://developer.nvidia.com/rdp/cudnn-download)，[Cublas 安装](https://developer.nvidia.com/cublas)，我们强烈建议您规范安装，统一到一个目录下，以免不必要的麻烦。

   > 如您的第三方加速卡为寒武纪 MLU，请参考寒武纪官方文档进行[驱动安装](https://www.cambricon.com/docs/sdk_1.11.0/driver_5.10.6/user_guide_5.10.6/index.html)，[CNToolkit 安装](https://www.cambricon.com/docs/sdk_1.11.0/cntoolkit_3.4.1/cntoolkit_install_3.4.1/index.html)，[CNNL 安装](https://www.cambricon.com/docs/sdk_1.11.0/cambricon_cnnl_1.16.1/user_guide/index.html)，我们强烈建议您规范安装，统一到一个目录下，以免不必要的麻烦。另外请注意，由于 MLU 上层软件建设适配程度有限，如您在其覆盖的机器，操作系统之外运行，需要在安装驱动之后使用上层软件的 Docker。

4. 确认您安装了 make，build-essential， python-is-python3， python-dev-is-python3， python3-pip， libdw-dev，如您的机器没有上述基础依赖，请自行按需安装。

   > 在使用 apt-get 工具情况下，您可以这样子执行。

   ```bash
   sudo apt-get install make cmake build-essential python-is-python3 python-dev-is-python3 python3-pip libdw-dev
   ```

   > 其他工具安装方式请自行上网搜寻

5. 更新pip并切换到清华源

   ```bash
   python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

## 编译本项目

推荐使用 X86-64 机器以及 Ubuntu-22.04，本文以此环境为例。

1. 编译本项目并打包成 Python 库进行安装

我们提供了意见编译参数，您可以在项目根目录下执行下面的命令。第一次执行会同时安装 python 依赖库，耗时略长，请耐心等待

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

2. 使用方法

安装成功后，您就可以使用本项目的 Python 接口进行编码并运行。具体使用方式可以参考项目样例代码 example/Resnet/resnet.py 以及用户使用手册

## 技术支持

如遇到问题，请联系我们技术支持团队