#include "cuda/cuda_common.h"
#include <cub/cub.cuh>

#define BLOCK_DIM_y 64
#define BLOCK_DIM_x 16

__global__ void _softmax_kernel(float *input, float *output, int size,
                                int size_y, int dimsize,
                                int stride) {      // if set axis = 1
    int i = threadIdx.x + blockIdx.x * blockDim.x; // i < inputShape[axis]
    int j = threadIdx.y + blockIdx.y * blockDim.y; // j < size/inputShape[axis]

    __shared__ float res_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float res_max[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float share_input[BLOCK_DIM_x][BLOCK_DIM_y];

    int tid = j % stride + (j - j % stride) * dimsize;
    if (i < dimsize && j < size_y) {
        share_input[threadIdx.x][threadIdx.y] = input[tid + i * stride];
    } else {
        share_input[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    }

    res_sum[threadIdx.x][threadIdx.y] = 0.0f;
    res_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    //__syncthreads();
    for (int ph = 0; threadIdx.x + ph * blockDim.x < dimsize; ph++) {
        if (ph == blockIdx.x) {
            if (res_max[threadIdx.x][threadIdx.y] >
                share_input[threadIdx.x][threadIdx.y]) {
                res_sum[threadIdx.x][threadIdx.y] =
                    res_sum[threadIdx.x][threadIdx.y] +
                    __expf(share_input[threadIdx.x][threadIdx.y] -
                           res_max[threadIdx.x][threadIdx.y]);

            } else {
                res_sum[threadIdx.x][threadIdx.y] =
                    1 + res_sum[threadIdx.x][threadIdx.y] *
                            __expf(res_max[threadIdx.x][threadIdx.y] -
                                   share_input[threadIdx.x][threadIdx.y]);
                res_max[threadIdx.x][threadIdx.y] =
                    share_input[threadIdx.x][threadIdx.y];
            }
        } else {
            if (res_max[threadIdx.x][threadIdx.y] >
                input[tid + (threadIdx.x + ph * blockDim.x) * stride]) {
                res_sum[threadIdx.x][threadIdx.y] =
                    res_sum[threadIdx.x][threadIdx.y] +
                    __expf(
                        input[tid + (threadIdx.x + ph * blockDim.x) * stride] -
                        res_max[threadIdx.x][threadIdx.y]);

            } else {
                res_sum[threadIdx.x][threadIdx.y] =
                    1 + res_sum[threadIdx.x][threadIdx.y] *
                            __expf(res_max[threadIdx.x][threadIdx.y] -
                                   input[tid + (threadIdx.x + ph * blockDim.x) *
                                                   stride]);
                res_max[threadIdx.x][threadIdx.y] =
                    input[tid + (threadIdx.x + ph * blockDim.x) * stride];
            }
        }
    }
    __syncthreads();
    for (int strip = blockDim.x / 2; strip > 0; strip = strip / 2) {
        if (threadIdx.x < strip) {
            if (res_max[threadIdx.x][threadIdx.y] >
                res_max[threadIdx.x + strip][threadIdx.y]) {
                res_sum[threadIdx.x][threadIdx.y] =
                    res_sum[threadIdx.x][threadIdx.y] +
                    res_sum[threadIdx.x + strip][threadIdx.y] *
                        __expf(res_max[threadIdx.x + strip][threadIdx.y] -
                               res_max[threadIdx.x][threadIdx.y]);
            } else {
                res_sum[threadIdx.x][threadIdx.y] =
                    res_sum[threadIdx.x + strip][threadIdx.y] +
                    res_sum[threadIdx.x][threadIdx.y] *
                        __expf(res_max[threadIdx.x][threadIdx.y] -
                               res_max[threadIdx.x + strip][threadIdx.y]);
                res_max[threadIdx.x][threadIdx.y] =
                    res_max[threadIdx.x + strip][threadIdx.y];
            }
        }
    }

    __syncthreads();
    //-----------------

    if (i < dimsize && j < size_y) {
        output[tid + i * stride] =
            __expf(share_input[threadIdx.x][threadIdx.y] -
                   res_max[0][threadIdx.y]) *
            __fdividef(1.0F, res_sum[0][threadIdx.y]);
    }
}
namespace infini {
void softmax_kernel(float *input, float *output, int size, int size_y,
                    int dimsize, int stride) {

    int num_block_x = ceil(dimsize / (double)BLOCK_DIM_x);
    int num_block_y = ceil(size_y / (double)(BLOCK_DIM_y));
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y);
    dim3 grid_dim(num_block_x, num_block_y);
    int share_mem = (3 * BLOCK_DIM_x * BLOCK_DIM_y) * sizeof(float);
    _softmax_kernel<<<grid_dim, block_dim, share_mem>>>(
        input, output, size, size_y, dimsize, stride);
}
} // namespace infini
