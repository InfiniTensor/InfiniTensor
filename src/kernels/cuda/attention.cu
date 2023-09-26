#include "cuda/cuda_common.h"

#define BLOCK_DIM_x 8 // BLOCK_DIM_x must <= 32
#define BLOCK_DIM_y 128
#define max_function(a, b) ((a) > (b) ? (a) : (b))

__global__ void _attentionKernel(const float *inputQ, const float *inputK,
                                 const float *inputV, int N, int d,
                                 float *output) {
    int i = blockIdx.x;                              // i must < N,Q[i]
    int phd = threadIdx.y + blockIdx.y * blockDim.y; // V[:,d]
    int phNumN = (N + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    __shared__ float old_max[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float new_max[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float new_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    old_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    new_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    new_sum[threadIdx.x][threadIdx.y] = 0.0f;
    __shared__ float block_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float block_max[BLOCK_DIM_x][BLOCK_DIM_y];
    block_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    block_sum[threadIdx.x][threadIdx.y] = 0.0f;

    __shared__ float inputS[BLOCK_DIM_x][BLOCK_DIM_y];

    __syncthreads();
    for (int phn = 0; phn < phNumN; phn++) {
        int j = threadIdx.x + phn * BLOCK_DIM_x;
        inputS[threadIdx.x][threadIdx.y] = 0.0f;
        block_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
        block_sum[threadIdx.x][threadIdx.y] = 0.0f;

        if (j < N && phd < d) {
            float sum_s = 0;
            for (int index = 0; index < d; index++) {
                sum_s += inputQ[i * d + index] * inputK[j * d + index];
            }
            inputS[threadIdx.x][threadIdx.y] = sum_s;
            block_max[threadIdx.x][threadIdx.y] = sum_s;
            block_sum[threadIdx.x][threadIdx.y] = 1.0f;
        }

        __syncthreads();
        for (int strip = BLOCK_DIM_x / 2; strip > 0; strip = strip / 2) {
            if (threadIdx.x < strip) {
                if (block_max[threadIdx.x][threadIdx.y] >
                    block_max[threadIdx.x + strip][threadIdx.y]) {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y] +
                        block_sum[threadIdx.x + strip][threadIdx.y] *
                            __expf(block_max[threadIdx.x + strip][threadIdx.y] -
                                   block_max[threadIdx.x][threadIdx.y]);
                } else {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x + strip][threadIdx.y] +
                        block_sum[threadIdx.x][threadIdx.y] *
                            __expf(block_max[threadIdx.x][threadIdx.y] -
                                   block_max[threadIdx.x + strip][threadIdx.y]);
                    block_max[threadIdx.x][threadIdx.y] =
                        block_max[threadIdx.x + strip][threadIdx.y];
                }
            }
            __syncthreads();
        }
        __syncthreads();
        if (j < N && phd < d) {
            if (new_max[threadIdx.x][threadIdx.y] > block_max[0][threadIdx.y]) {
                new_sum[threadIdx.x][threadIdx.y] =
                    new_sum[threadIdx.x][threadIdx.y] +
                    block_sum[0][threadIdx.y] *
                        __expf(block_max[0][threadIdx.y] -
                               new_max[threadIdx.x][threadIdx.y]);
            } else {
                new_sum[threadIdx.x][threadIdx.y] =
                    block_sum[0][threadIdx.y] +
                    new_sum[threadIdx.x][threadIdx.y] *
                        __expf(new_max[threadIdx.x][threadIdx.y] -
                               block_max[0][threadIdx.y]);
                new_max[threadIdx.x][threadIdx.y] = block_max[0][threadIdx.y];
            }
        }

        __syncthreads();

        if (j < N && phd < d) {
            inputS[threadIdx.x][threadIdx.y] =
                __expf(inputS[threadIdx.x][threadIdx.y] -
                       new_max[threadIdx.x][threadIdx.y]);
        } else {
            inputS[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();

        if (phd < d) {
            float sum_o = 0.0f;
            for (int index = 0; index < BLOCK_DIM_x; index++) {
                if (index + phn * BLOCK_DIM_x < N) {
                    sum_o += inputS[index][threadIdx.y] *
                             inputV[(index + phn * BLOCK_DIM_x) * d + phd];
                }
            }
            if (phn == 0) {
                output[i * d + phd] = sum_o;
            } else {
                output[i * d + phd] =
                    __expf(old_max[threadIdx.x][threadIdx.y] -
                           new_max[threadIdx.x][threadIdx.y]) *
                        output[i * d + phd] +
                    sum_o;
            }

            old_max[threadIdx.x][threadIdx.y] =
                new_max[threadIdx.x][threadIdx.y];
        } else {
            old_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
        }
        __syncthreads();
    }
    __syncthreads();
    if (phd < d)
        output[i * d + phd] =
            output[i * d + phd] *
            __fdividef(1.0F, new_sum[threadIdx.x][threadIdx.y]);
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {

    int num_block_x = N;
    int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    int share_mem =
        (3 * BLOCK_DIM_x + 3 * BLOCK_DIM_x) * BLOCK_DIM_y * sizeof(float);
    _attentionKernel<<<grid_dim, block_dim, share_mem>>>(inputQ, inputK, inputV,
                                                         N, d, output);
}
} // namespace infini