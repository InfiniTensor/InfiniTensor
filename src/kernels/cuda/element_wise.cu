#include "cuda/cuda_common.h"
#include <math.h>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _div_kernel(float *x, float *y, float *z, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        z[i] = x[i] / y[i];
    }
}

__global__ void _pow_kernel(float *x, float *y, float *z, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        z[i] = pow(x[i], y[i]);
    }
}

namespace infini {
void div_kernel(float *a, float *b, float *c, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _div_kernel<<<blocksize, gridsize>>>(a, b, c, num);
}
void pow_kernel(float *a, float *b, float *c, int num) {

    int blocksize = block_work_size();
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _pow_kernel<<<blocksize, gridsize>>>(a, b, c, num);
}

}; // namespace infini
