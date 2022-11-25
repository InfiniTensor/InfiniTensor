#include "cuda/cuda_common.h"
#include <cstdio>

__global__ void cudaPrintFloatImpl(float *x, int len) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    if (start == 0) {
        for (int i = 0; i < len; ++i) {
            printf("%.7f ", x[i]);
        }
        printf("\n");
    }
}

namespace infini {

void cudaPrintFloat(float *x, int len) {
    cudaPrintFloatImpl<<<1, 1>>>(x, len);
    cudaDeviceSynchronize();
}

} // namespace infini
