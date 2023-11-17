#include "cuda/cuda_common.h"

template <int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output) {
    int i = blockIdx.y;                              // i must < N,Q[i]
    int phd = threadIdx.x + blockIdx.x * blockDim.x; // V[:,d]

    int phNumN = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    __shared__ float inputS[BLOCK_DIM_x][BLOCK_DIM_y];
    float newMax;
    float oldMax;
    float newSum;

    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 0.0f;

    float out;
    out = 0.0f;
    //---------
    __shared__ float block_sum[BLOCK_DIM_x][BLOCK_DIM_y];

    __shared__ float sum_partial[BLOCK_DIM_x][BLOCK_DIM_y];
    int extra = d % BLOCK_DIM_x;
    int step = (d - extra) / BLOCK_DIM_x;
    for (int phn = 0; phn < phNumN; phn++) {

        int j = threadIdx.y + phn * BLOCK_DIM_y;

        float sum_r = 0.0f;
        __syncthreads();
        if (threadIdx.x < extra) {
            for (int ind = threadIdx.x * (step + 1);
                 ind < (threadIdx.x + 1) * (step + 1); ind++) {
                sum_r += inputQ[i * d + ind] * inputK[j * d + ind];
            }
        } else {
            for (int ind = extra * (step + 1) + (threadIdx.x - extra) * step;
                 ind < extra * (step + 1) + (threadIdx.x - extra + 1) * step;
                 ind++) {
                sum_r += inputQ[i * d + ind] * inputK[j * d + ind];
            }
        }
        if (j < N) {
            sum_partial[threadIdx.x][threadIdx.y] = sum_r;
        } else {
            sum_partial[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int strip = BLOCK_DIM_x / 2; strip > 0; strip /= 2) {
            if (threadIdx.x < strip) {
                sum_partial[threadIdx.x][threadIdx.y] +=
                    sum_partial[threadIdx.x + strip][threadIdx.y];
            }
            __syncthreads();
        }
        float sum_s = sum_partial[0][threadIdx.y];
        if (j < N) {

            block_sum[threadIdx.x][threadIdx.y] = 1.0f;
        } else {

            sum_partial[0][threadIdx.y] = -__FLT_MAX__;
            block_sum[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int strip = BLOCK_DIM_y / 2; strip > 0; strip /= 2) {
            if (threadIdx.y < strip) {
                if (sum_partial[0][threadIdx.y] >
                    sum_partial[0][threadIdx.y + strip]) {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y] +
                        block_sum[threadIdx.x][threadIdx.y + strip] *
                            __expf(sum_partial[0][threadIdx.y + strip] -
                                   sum_partial[0][threadIdx.y]);
                } else {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y + strip] +
                        block_sum[threadIdx.x][threadIdx.y] *
                            __expf(sum_partial[0][threadIdx.y] -
                                   sum_partial[0][threadIdx.y + strip]);
                    sum_partial[0][threadIdx.y] =
                        sum_partial[0][threadIdx.y + strip];
                }
            }
            __syncthreads();
        }
        if (newMax > sum_partial[0][0]) {
            newSum = newSum + block_sum[threadIdx.x][0] *
                                  __expf(sum_partial[0][0] - newMax);
        } else {
            newSum = block_sum[threadIdx.x][0] +
                     newSum * __expf(newMax - sum_partial[0][0]);
            newMax = sum_partial[0][0];
        }

        if (j < N && phd < d) {
            inputS[threadIdx.x][threadIdx.y] =
                __expf(sum_s - newMax) *
                inputV[(threadIdx.y + phn * BLOCK_DIM_y) * d + phd];
        } else {
            inputS[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int strip = BLOCK_DIM_y / 2; strip > 0; strip /= 2) {
            if (threadIdx.y < strip) {
                inputS[threadIdx.x][threadIdx.y] +=
                    inputS[threadIdx.x][threadIdx.y + strip];
            }
            __syncthreads();
        }
        if (j < N && phd < d) {
            out = __expf(oldMax - newMax) * out + inputS[threadIdx.x][0];
        }
        oldMax = newMax;
    }

    if (threadIdx.y + (phNumN - 1) * BLOCK_DIM_y < N && phd < d) {
        output[i * d + phd] = out * __fdividef(1.0F, newSum);
    }
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {
    int num_block_y = N;
    if (d > 512) {
        int BLOCK_DIM_x = 1024;
        int BLOCK_DIM_y = 1;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<1024, 1>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 256) {
        int BLOCK_DIM_x = 512;
        int BLOCK_DIM_y = 2;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<512, 2>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 128) {
        int BLOCK_DIM_x = 256;
        int BLOCK_DIM_y = 4;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<256, 4>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 64) {
        int BLOCK_DIM_x = 128;
        int BLOCK_DIM_y = 8;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<128, 8>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 32) {
        int BLOCK_DIM_x = 64;
        int BLOCK_DIM_y = 16;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<64, 16>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 16) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<32, 32>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<16, 64>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    }
}
} // namespace infini
