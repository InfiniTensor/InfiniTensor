#include "cuda/cuda_common.h"
#include <cub/block/block_reduce.cuh>
#define max_function(a, b) ((a) > (b) ? (a) : (b))

template <int BLOCK_DIM_x>
__launch_bounds__(BLOCK_DIM_x) __global__
    void _attentionKernel(const float *__restrict inputQ,
                          const float *__restrict inputK,
                          const float *__restrict inputV, int N, int d,
                          float *__restrict output) {
    int i = blockIdx.y;                              // i must < N,Q[i]
    int phd = threadIdx.x + blockIdx.x * blockDim.x; // V[:,d]

    float old_max = -__FLT_MAX__;
    float new_max = -__FLT_MAX__;
    float new_sum = 0.0f;

    __shared__ float out[BLOCK_DIM_x];

    int extra = d % BLOCK_DIM_x;
    int step = (d - extra) / BLOCK_DIM_x;
    out[threadIdx.x] = 0.0f;
    __shared__ float sum_s;
    for (int phn = 0; phn < N; phn++) {
        float sum_partial = 0.0f;

        if (threadIdx.x < extra) {
            for (int ind = threadIdx.x * (step + 1);
                 ind < (threadIdx.x + 1) * (step + 1); ind++) {
                sum_partial += inputQ[i * d + ind] * inputK[phn * d + ind];
            }
        } else {
            for (int ind = extra * (step + 1) + (threadIdx.x - extra) * step;
                 ind < extra * (step + 1) + (threadIdx.x - extra + 1) * step;
                 ind++) {
                sum_partial += inputQ[i * d + ind] * inputK[phn * d + ind];
            }
        }
        typedef cub::BlockReduce<float, BLOCK_DIM_x> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float block_sum =
            BlockReduce(temp_storage).Reduce(sum_partial, cub::Sum());

        if (threadIdx.x == 0)
            sum_s = block_sum;
        __syncthreads();

        if (new_max > sum_s) {
            new_sum = new_sum + __expf(sum_s - new_max);
        } else {
            new_sum = 1.0f + new_sum * __expf(new_max - sum_s);
            new_max = sum_s;
        }

        sum_s = __expf(sum_s - new_max);

        out[threadIdx.x] = __expf(old_max - new_max) * out[threadIdx.x] +
                           sum_s * inputV[phn * d + phd];

        old_max = new_max;
    }

    if (phd < d)
        output[i * d + phd] = out[threadIdx.x] * __fdividef(1.0F, new_sum);
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {

    int num_block_y = N;

    if (d > 512) {
        int BLOCK_DIM_x = 1024;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<1024>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 256) {
        int BLOCK_DIM_x = 512;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<512>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 128) {
        int BLOCK_DIM_x = 256;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<256>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 64) {
        int BLOCK_DIM_x = 128;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<128>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 32) {
        int BLOCK_DIM_x = 64;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<512>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 16) {
        int BLOCK_DIM_x = 32;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<32>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else {
        int BLOCK_DIM_x = 16;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<16>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    }
}
} // namespace infini