. /home/spack/spack/share/spack/setup-env.sh
spack load cuda@11.0.2 cudnn@8.0.3.33-11.0 intel-oneapi-dnn@2022.1.0 intel-oneapi-mkl@2022.1.0 
export CUDAHOSTCXX=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin/gcc
# The default dnnl library is cpu_dpcpp_gpu_dpcpp which requires libsycl.so, after "spack load", and need to change to gomp explicitly.
export LD_LIBRARY_PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-12.1.0/intel-oneapi-dnn-2022.1.0-7rs6ht57zozyxhxx6s2qlrqzmqknhgzx/dnnl/2022.1.0/cpu_gomp/lib/:$LD_LIBRARY_PATH
