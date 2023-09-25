#include "cuda/cuda_common.h"

#define BLOCK_DIM_x 2
#define BLOCK_DIM_y 2
#define max_function(a, b) ((a) > (b) ? (a) : (b))

__global__ void _attentionKernel(const float *inputQ, const float *inputK,
                                 const float *inputV, int N, int d,
                                 float *output) {
    int i = blockIdx.x;                              // i must < N,Q[i]
    int phd = threadIdx.y + blockIdx.y * blockDim.y; // V[:,d]
    int phNumN = (N + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    __shared__ float old_max;
    __shared__ float new_max;
    __shared__ float new_sum;
    old_max = -__FLT_MAX__;
    new_max = -__FLT_MAX__;
    new_sum = 0.0f;
    __shared__ float block_sum[BLOCK_DIM_x];
    __shared__ float block_max[BLOCK_DIM_x];
    block_max[threadIdx.x] = -__FLT_MAX__;
    block_sum[threadIdx.x] = 0.0f;

    __shared__ float inputS[BLOCK_DIM_x];

    output[i * d + phd] = 0.0f;
    for (int phn = 0; phn < phNumN; phn++) {
        int j = threadIdx.x + phn * BLOCK_DIM_x;
        if (j < N) {
            float sum_s = 0;
            for (int index = 0; index < d; index++) {
                sum_s += inputQ[i * d + index] * inputK[j * d + index];
            }
            inputS[threadIdx.x] = sum_s;
            block_max[threadIdx.x] = sum_s;
            block_sum[threadIdx.x] = 1.0f;
        } else {
            inputS[threadIdx.x] = 0.0f;
            block_max[threadIdx.x] = -__FLT_MAX__;
            block_sum[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int strip = BLOCK_DIM_x / 2; strip > 0; strip = strip / 2) {
            if (threadIdx.x < strip) {
                if (block_max[threadIdx.x] > block_max[threadIdx.x + strip]) {
                    block_sum[threadIdx.x] =
                        block_sum[threadIdx.x] +
                        block_sum[threadIdx.x + strip] *
                            __expf(block_max[threadIdx.x + strip] -
                                   block_max[threadIdx.x]);
                } else {
                    block_sum[threadIdx.x] =
                        block_sum[threadIdx.x + strip] +
                        block_sum[threadIdx.x] *
                            __expf(block_max[threadIdx.x] -
                                   block_max[threadIdx.x + strip]);
                    block_max[threadIdx.x] = block_max[threadIdx.x + strip];
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (new_max > block_max[0]) {
                new_sum =
                    new_sum + block_sum[0] * __expf(block_max[0] - new_max);
            } else {
                new_sum =
                    block_sum[0] + new_sum * __expf(new_max - block_max[0]);
                new_max = block_max[0];
            }
        }
        __syncthreads();
        inputS[threadIdx.x] = __expf(inputS[threadIdx.x] - new_max);
        __syncthreads();
        float sum_o = 0;
        if (phd < d) {
            for (int index = 0; index < BLOCK_DIM_x; index++) {
                if (index + phn * BLOCK_DIM_x < N) {
                    sum_o += inputS[index] *
                             inputV[(index + phn * BLOCK_DIM_x) * d + phd];
                }
            }
            output[i * d + phd] =
                __expf(old_max - new_max) * output[i * d + phd] + sum_o;
            old_max = new_max;
        }
        //__syncthreads();
    }
    if (phd < d)
        output[i * d + phd] = output[i * d + phd] * __fdividef(1.0F, new_sum);
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {

    int num_block_x = N;
    int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    int share_mem = (3 * BLOCK_DIM_x + 3) * sizeof(float);
    _attentionKernel<<<grid_dim, block_dim, share_mem>>>(inputQ, inputK, inputV,
                                                         N, d, output);
}
} // namespace infini