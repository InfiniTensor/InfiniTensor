# MetaX MACA (matches vllm-metax / infinilm-prefill-dev image defaults)
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUCC_PATH="${MACA_PATH}/tools/cu-bridge"
export CUDA_PATH="${CUCC_PATH}"
export PATH="${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${CUCC_PATH}/bin:${PATH}"
export CPATH="${MACA_PATH}/tools/cu-bridge/include:${MACA_PATH}/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${MACA_PATH}/tools/cu-bridge/include:${MACA_PATH}/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="${MACA_PATH}/tools/cu-bridge/include:${MACA_PATH}/include:${C_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/tools/cu-bridge/lib:${LD_LIBRARY_PATH:-}"


# tianshu
export COREX_PATH="/usr/local/corex"
# 添加编译器路径
export PATH="${COREX_PATH}/bin:${PATH}"
export PATH="/usr/local/bin:${PATH}"
# 添加动态库路径  
export LD_LIBRARY_PATH="${COREX_PATH}/lib64:${LD_LIBRARY_PATH}"