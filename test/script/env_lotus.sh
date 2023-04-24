#!/bin/bash

. /home/spack/spack/share/spack/setup-env.sh
if [ "$#" == 0 ] || [ "$1" == "cuda" ]
then
    echo "Load CUDA environment."
    spack load cuda@11.0.2 cudnn@8.0.3.33-11.0 
    spack load /2fcgebh # python3.7
    export CUDAHOSTCXX=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin/gcc
elif [ "$1" == "intelcpu" ]
then
    echo "Load INTELCPU environment."
    spack load gcc@12.1.0 intel-oneapi-dnn@2022.1.0 intel-oneapi-mkl@2022.1.0 intel-oneapi-compilers@2022.1.0
    # The default dnnl library is cpu_dpcpp_gpu_dpcpp which requires libsycl.so, after "spack load", and need to change to gomp explicitly.
    export LD_LIBRARY_PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-12.1.0/intel-oneapi-dnn-2022.1.0-7rs6ht57zozyxhxx6s2qlrqzmqknhgzx/dnnl/2022.1.0/cpu_gomp/lib/:$LD_LIBRARY_PATH


    #  flopen mkl libs will fail when used by python. 
    #  Refering to "https://groups.google.com/g/kaldi-help/c/m3nyQke0HS0/m/4fj8gkSWAgAJ", it is recommended to use mkl_rt instead,
    #  but mkl_rt do not support dpc++ refered to https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/using-the-single-dynamic-library.html
    #  Preloading the missing libs will work, refered to https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/mkl-fails-to-load/m-p/1155538

    export MKLLIB_PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-12.1.0/intel-oneapi-mkl-2022.1.0-mf6te62fo6wxlo33jwwwgg5kljoagc6g/mkl/2022.1.0/
    export LD_PRELOAD=$MKLLIB_PATH/lib/intel64/libmkl_def.so.2:$MKLLIB_PATH/lib/intel64/libmkl_avx2.so.2:$MKLLIB_PATH/lib/intel64/libmkl_core.so:$MKLLIB_PATH/lib/intel64/libmkl_intel_lp64.so:$MKLLIB_PATH/lib/intel64/libmkl_intel_thread.so:/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-12.1.0/intel-oneapi-compilers-2022.1.0-6k6zm3h4qcsni27nihc4b6wuqgtxqxqa/compiler/2022.1.0/linux/compiler/lib/intel64_lin/libiomp5.so
else
    echo "Bad option. Please enter 'cuda' or 'intelcpu'. CUDA will be loaded by default if nothing specified."
fi
