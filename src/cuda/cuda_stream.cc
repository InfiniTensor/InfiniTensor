#include "cuda/cuda_common.h"

namespace infini {
std::unique_ptr<CUDAStream> CUDAStream::p_CUDAStream;
CUDAStream::CUDAStream() {}

} // namespace infini
