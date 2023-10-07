#include "cuda/cuda_common.h"

#define max_function(a, b) ((a) > (b) ? (a) : (b))

template <int BLOCK_DIM_y>
__launch_bounds__(BLOCK_DIM_y) __global__
    void _attentionKernel(const float *__restrict inputQ,
                          const float *__restrict inputK,
                          const float *__restrict inputV, int N, int d,
                          float *__restrict output) {
    int i = blockIdx.x;                              // i must < N,Q[i]
    int phd = threadIdx.y + blockIdx.y * blockDim.y; // V[:,d]

    float old_max = -__FLT_MAX__;
    float new_max = -__FLT_MAX__;
    float new_sum = 0.0f;

    __shared__ float out[BLOCK_DIM_y];

    int extra = d % BLOCK_DIM_y;
    int step = (d - extra) / BLOCK_DIM_y;
    __shared__ float shareQ_times_K[BLOCK_DIM_y];

    for (int phn = 0; phn < N; phn++) {
        shareQ_times_K[threadIdx.y] = 0.0f;
        float sum_s = 0.0f;
        if (threadIdx.y < extra) {
            for (int ind = threadIdx.y * (step + 1);
                 ind < (threadIdx.y + 1) * (step + 1); ind++) {
                shareQ_times_K[threadIdx.y] +=
                    inputQ[i * d + ind] * inputK[phn * d + ind];
            }
        } else {
            for (int ind = extra * (step + 1) + (threadIdx.y - extra) * step;
                 ind < extra * (step + 1) + (threadIdx.y - extra + 1) * step;
                 ind++) {
                shareQ_times_K[threadIdx.y] +=
                    inputQ[i * d + ind] * inputK[phn * d + ind];
            }
        }

        __syncthreads();
        for (int strip = BLOCK_DIM_y / 8; strip > 0; strip = strip / 8) {
            if (threadIdx.y < strip) {
                for (int id = 1; id < 8; id++) {
                    shareQ_times_K[threadIdx.y] +=
                        shareQ_times_K[threadIdx.y + id * strip];
                }
            }
            __syncthreads();
        }
        sum_s = shareQ_times_K[0] + shareQ_times_K[1];
        //__syncthreads();

        if (new_max > sum_s) {
            new_sum = new_sum + __expf(sum_s - new_max);
        } else {
            new_sum = 1.0f + new_sum * __expf(new_max - sum_s);
            new_max = sum_s;
        }

        //__syncthreads();

        sum_s = __expf(sum_s - new_max);

        //__syncthreads();

        if (phn == 0) {
            out[threadIdx.y] = sum_s * inputV[phn * d + phd];

        } else {
            out[threadIdx.y] = __expf(old_max - new_max) * out[threadIdx.y] +
                               sum_s * inputV[phn * d + phd];
        }

        old_max = new_max;

        //__syncthreads();
    }
    //__syncthreads();
    if (phd < d)
        output[i * d + phd] = out[threadIdx.y] * __fdividef(1.0F, new_sum);
}
namespace infini {
void attentionKernel(const float *inputQ, const float *inputK,
                     const float *inputV, int N, int d, float *output) {

    int num_block_x = N;

    if (d > 128) {
        int BLOCK_DIM_y = 1024;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<1024>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 16) {
        int BLOCK_DIM_y = 128;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<128>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else {
        int BLOCK_DIM_y = 16;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<16>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    }
}
} // namespace infini
