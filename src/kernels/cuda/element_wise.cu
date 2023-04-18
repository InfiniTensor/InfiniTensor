#include "cuda/cuda_common.h"
#include <math.h>

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _div_kernel(float *x, float *y, float *z, int a0, int a1, int a2, int a3,
                                                          int b0, int b1, int b2, int b3,
                                                          int c0, int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i/ (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        z[i] = x[a0_index*a1*a2*a3 + a1_index*a2*a3 + a2_index*a3 + a3_index] / y[b0_index*b1*b2*b3 + b1_index*b2*b3 + b2_index*b3 + b3_index];
    }
}

__global__ void _pow_kernel(float *x, float *y, float *z, int a0, int a1, int a2, int a3,
                                                          int b0, int b1, int b2, int b3,
                                                          int c0, int c1, int c2, int c3) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n = c0 * c1 * c2 * c3;

    for (int i = index; i < n; i += stride) {
        int c0_index = i/ (c1 * c2 * c3);
        int c1_index = (i % (c1 * c2 * c3)) / (c2 * c3);
        int c2_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) / c3;
        int c3_index = ((i % (c1 * c2 * c3)) % (c2 * c3)) % c3;

        int a0_index = c0_index % a0;
        int a1_index = c1_index % a1;
        int a2_index = c2_index % a2;
        int a3_index = c3_index % a3;

        int b0_index = c0_index % b0;
        int b1_index = c1_index % b1;
        int b2_index = c2_index % b2;
        int b3_index = c3_index % b3;
        z[i] = pow(x[a0_index*a1*a2*a3 + a1_index*a2*a3 + a2_index*a3 + a3_index], y[b0_index*b1*b2*b3 + b1_index*b2*b3 + b2_index*b3 + b3_index]);
    }
}

namespace infini {
void div_kernel(float *a, float *b, float *c, int a0, int a1, int a2, int a3,
                                              int b0, int b1, int b2, int b3,
                                              int c0, int c1, int c2, int c3) {

    int blocksize = block_work_size();
    int num = c0*c1*c2*c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _div_kernel<<<blocksize, gridsize>>>(a, b, c, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);
}
void pow_kernel(float *a, float *b, float *c, int a0, int a1, int a2, int a3,
                                              int b0, int b1, int b2, int b3,
                                              int c0, int c1, int c2, int c3) {
    int blocksize = block_work_size();
    int num = c0*c1*c2*c3;
    int gridsize = (num + block_work_size() - 1) / block_work_size();
    _pow_kernel<<<blocksize, gridsize>>>(a, b, c, a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3);
}

}; // namespace infini
