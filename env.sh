# 配置英伟达 CUDA 的 HOME 路径，请注意安装 CUDA Toolkit, CUDNN 并将路径配置到下述环境变量。
export CUDA_HOME=/PATH/TO/YOUR/CUDA/HOME
export CUDNN_HOME=/PATH/TO/YOUR/CUDNN/HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 配置寒武纪 BANG 的 HOME 路径，请注意 /usr/local/neuware 是寒武纪软件栈建议的，同时也是默认的安装路径。
# 如若用户有其他的路径安装方式，请自行配置正确的路径。
# 这里是 neuware 目录下一个可能的结构图，请参考。
# .
# ├── bin
# ├── cmake
# ├── data
# ├── edge
# ├── include
# ├── lib
# ├── lib64
# ├── LICENSE
# ├── mlvm
# ├── README
# ├── samples
# ├── share
# └── version.txt
export NEUWARE_HOME=/usr/local/neuware
export PATH="${NEUWARE_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH}"

# 配置昆仑芯 XPU 的 HOME 路径，请注意 /usr/local/xpu 是昆仑芯软件栈提供的软件包路径。
# 如若用户有其他的路径安装方式，请自行配置正确的路径。
# 这里是 xpu 目录下一个可能的结构图，请参考。
# .
# ├── bin
# ├── include
# ├── lib64
# ├── tools
# ├── version
# └── XTDK
export KUNLUN_HOME=/usr/local/xpu

# 配置华为ASCEND NPU 的 HOME 路径，请注意 /usr/local/Ascend/ascend-toolkit/latest 是华为ASCEND 软件栈提供的软件包路径。
# 如若用户有其他的路径安装方式，请自行配置正确的路径。
# 这里是 ascend 目录下一个可能的结构图，请参考。
# .
# ├── aarch64-linux
# ├── acllib
# ├── arm64-linux
# ├── atc
# ├── bin
# ├── compiler
# ├── conf
# ├── fwkacllib
# ├── hccl
# ├── include
# ├── lib64
# ├── mindstudio-toolkit
# ├── opp
# ├── opp_kernel
# ├── ops
# ├── pyACL
# ├── python
# ├── runtime
# ├── test-ops
# ├── toolkit
# └── tools

export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/toolbox/set_env.sh
