#include "cuda/cuda_common.h"

#define BLOCK_DIM_x 2
#define BLOCK_DIM_y 2

#define max_function(a, b) ((a) > (b) ? (a) : (b))

__global__ void _attentionKernel(const float *inputQ, const float *inputK,
                                 const float *inputV, int N, int d,
                                 float *output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; //
    int phNumN = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;

    __shared__ float block_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float block_max[BLOCK_DIM_x][BLOCK_DIM_y];
    block_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    block_sum[threadIdx.x][threadIdx.y] = 0.0f;
    __shared__ float grid_sum[BLOCK_DIM_x];
    __shared__ float grid_max[BLOCK_DIM_x];
    __shared__ float grid_max_old[BLOCK_DIM_x];
    grid_max[threadIdx.x] = -__FLT_MAX__;
    grid_max_old[threadIdx.x] = -__FLT_MAX__;
    grid_sum[threadIdx.x] = 0.0f;
    __shared__ float S[BLOCK_DIM_x][BLOCK_DIM_y];

    __shared__ float Out_new[BLOCK_DIM_x][BLOCK_DIM_y];
    Out_new[threadIdx.x][threadIdx.y] = 0.0f;
    for (int phn = 0; phn < phNumN; phn++) {
        int j = threadIdx.y + phn * BLOCK_DIM_y;
        if (i < N && j < N) {
            float sum_s = 0;
            for (int index = 0; index < d; index++) {
                sum_s += inputQ[i * d + index] * inputK[j * d + index];
            }

            S[threadIdx.x][threadIdx.y] = sum_s;
            block_sum[threadIdx.x][threadIdx.y] = 1.0f;
            block_max[threadIdx.x][threadIdx.y] = sum_s;
        } else {
            S[threadIdx.x][threadIdx.y] = 0.0f;
            block_sum[threadIdx.x][threadIdx.y] = 0.0f;
            block_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
        }

        //----------------fix i, compute the max S[i,j] of this block
        __syncthreads();

        for (int strip = BLOCK_DIM_y / 2; strip > 0; strip = strip / 2) {
            if (threadIdx.y < strip) {
                if (block_max[threadIdx.x][threadIdx.y] >
                    block_max[threadIdx.x][threadIdx.y + strip]) {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y] +
                        block_sum[threadIdx.x][threadIdx.y + strip] *
                            __expf(block_max[threadIdx.x][threadIdx.y + strip] -
                                   block_max[threadIdx.x][threadIdx.y]);
                } else {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y + strip] +
                        block_sum[threadIdx.x][threadIdx.y] *
                            __expf(block_max[threadIdx.x][threadIdx.y] -
                                   block_max[threadIdx.x][threadIdx.y + strip]);
                    block_max[threadIdx.x][threadIdx.y] =
                        block_max[threadIdx.x][threadIdx.y + strip];
                }
            }
        } // block_max[threadIdx.x][0]store the local max of this block
        __syncthreads();
        if (threadIdx.y == 0) {
            if (grid_max[threadIdx.x] > block_max[threadIdx.x][0]) {
                grid_sum[threadIdx.x] = grid_sum[threadIdx.x] +
                                        block_sum[threadIdx.x][0] *
                                            __expf(block_max[threadIdx.x][0] -
                                                   grid_max[threadIdx.x]);
            } else {
                grid_sum[threadIdx.x] =
                    block_sum[threadIdx.x][0] +
                    grid_sum[threadIdx.x] * __expf(grid_max[threadIdx.x] -
                                                   block_max[threadIdx.x][0]);
                grid_max[threadIdx.x] = block_max[threadIdx.x][0];
            } // compare the max between the different blocks, when the loop
              // end, grid_max store the global max
        }
        __syncthreads();

        S[threadIdx.x][threadIdx.y] =
            __expf(S[threadIdx.x][threadIdx.y] -
                   grid_max[threadIdx.x]); // softmax(s)*L

        __syncthreads();
        int vj = threadIdx.y + blockIdx.y * blockDim.y;
        // do not write vj = threadIdx.y + ph * blockDim.y
        float sum_o;
        if (vj < d) {
            sum_o = 0;
            for (int vid = 0; vid < BLOCK_DIM_y; vid++) {
                if (vid + phn * BLOCK_DIM_y < N) {
                    sum_o += S[threadIdx.x][vid] *
                             inputV[(vid + phn * BLOCK_DIM_y) * d + vj];
                }
            }
            Out_new[threadIdx.x][threadIdx.y] =
                __expf(grid_max_old[threadIdx.x] - grid_max[threadIdx.x]) *
                    Out_new[threadIdx.x][threadIdx.y] +
                sum_o;
            grid_max_old[threadIdx.x] = grid_max[threadIdx.x];
        }
    }
    __syncthreads();

    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < N && j < d) {

        output[i * d + j] = Out_new[threadIdx.x][threadIdx.y] *
                            __fdividef(1.0F, grid_sum[threadIdx.x]);
    }
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {
    int num_block_y = (max_function(N, d) + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    int num_block_x = (N + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int share_mem = (5 * BLOCK_DIM_y + 2) * BLOCK_DIM_x * sizeof(float);
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    _attentionKernel<<<grid_dim, block_dim, share_mem>>>(inputQ, inputK, inputV,
                                                         N, d, output);
}
} // namespace infini
