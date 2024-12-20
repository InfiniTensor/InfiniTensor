#pragma once

#include "cuda/cuda_runtime.h"
#include "operators/det.h"

namespace infini {
void det_kernel(const CudaRuntimeObj *context, void *input, void *output,
                const int n, const int batch_size, int mode);

}; // namespace infini
