#include "cuda/cuda_common.h"
#include <cub/cub.cuh>

template <int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void blockLaynormKernel(const float *input, const float *scale,
                            const float *bias, const float eps, int scaleSize,
                            int biasSize, const int dimsize, const int stride,
                            float *output) {
    // len(scale) = len(bias) = dimsize
    int tmp = blockIdx.x % stride;
    int tid = (blockIdx.x - tmp) * dimsize + tmp;
    float muPartial = 0.0f;
    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
        muPartial += input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride];
    }
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu;
    float muBlock = BlockReduce(temp_storage).Reduce(muPartial, cub::Sum());
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        mu = muBlock / dimsize;
    }
    __syncthreads();

    float sigma2Partial = 0.0f;
    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
        sigma2Partial +=
            (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu) *
            (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu);
    }
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;

    __shared__ float sigma2;
    float sigma2Block =
        BlockReduce(temp_storage).Reduce(sigma2Partial, cub::Sum());
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        sigma2 = sigma2Block / dimsize;
    }
    __syncthreads();
    if (bias == NULL) {
        if (scaleSize == dimsize) {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[threadIdx.x + ph * BLOCK_DIM] *
                    (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                     mu) /
                    sqrt(sigma2 + eps);
            }
        } else {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[0] *
                    (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                     mu) /
                    sqrt(sigma2 + eps);
            }
        }
    } else if (biasSize == dimsize) {
        if (scaleSize == dimsize) {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[threadIdx.x + ph * BLOCK_DIM] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) /
                        sqrt(sigma2 + eps) +
                    bias[threadIdx.x + ph * BLOCK_DIM];
            }
        } else {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[0] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) /
                        sqrt(sigma2 + eps) +
                    bias[threadIdx.x + ph * BLOCK_DIM];
            }
        }
    } else {
        if (scaleSize == dimsize) {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[threadIdx.x + ph * BLOCK_DIM] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) /
                        sqrt(sigma2 + eps) +
                    bias[0];
            }
        } else {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[0] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) /
                        sqrt(sigma2 + eps) +
                    bias[0];
            }
        }
    }
}
template <typename T> struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

template <template <typename> class ReductionOp, typename T,
          int thread_group_width>
__inline__ __device__ T WarpAllReduce(T val) {
    for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
template <int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void warpLaynormKernel(const float *input, const float *scale,
                                  const float *bias, const float eps,
                                  int scaleSize, int biasSize,
                                  const int dimsize, const int otherSize,
                                  const int stride, float *output) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    if (otherIdx < otherSize) {

        __shared__ float muTotal[BLOCK_DIM_y];
        __shared__ float sigma2Total[BLOCK_DIM_y];

        float muPartial = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            muPartial += input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride];
        }

        muPartial = WarpAllReduce<SumOp, float, BLOCK_DIM_x>(muPartial);

        if (threadIdx.x == 0)
            muTotal[threadIdx.y] = muPartial / dimsize;

        //--------------------------------------------
        float sigma2Partial = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            sigma2Partial +=
                (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                 muTotal[threadIdx.y]) *
                (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                 muTotal[threadIdx.y]);
        }

        sigma2Partial = WarpAllReduce<SumOp, float, BLOCK_DIM_x>(sigma2Partial);

        if (threadIdx.x == 0)
            sigma2Total[threadIdx.y] = sigma2Partial / dimsize;

        //--------------------------------------------
        if (bias == NULL) {
            if (scaleSize == dimsize) {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[threadIdx.x + ph * BLOCK_DIM_x] *
                        (input[tid +
                               (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                         muTotal[threadIdx.y]) /
                        sqrt(sigma2Total[threadIdx.y] + eps);
                }
            } else {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[0] *
                        (input[tid +
                               (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                         muTotal[threadIdx.y]) /
                        sqrt(sigma2Total[threadIdx.y] + eps);
                }
            }
        } else if (biasSize == dimsize) {
            if (scaleSize == dimsize) {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[threadIdx.x + ph * BLOCK_DIM_x] *
                            (input[tid +
                                   (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                             muTotal[threadIdx.y]) /
                            sqrt(sigma2Total[threadIdx.y] + eps) +
                        bias[threadIdx.x + ph * BLOCK_DIM_x];
                }
            } else {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[0] *
                            (input[tid +
                                   (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                             muTotal[threadIdx.y]) /
                            sqrt(sigma2Total[threadIdx.y] + eps) +
                        bias[threadIdx.x + ph * BLOCK_DIM_x];
                }
            }
        } else {
            if (scaleSize == dimsize) {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[threadIdx.x + ph * BLOCK_DIM_x] *
                            (input[tid +
                                   (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                             muTotal[threadIdx.y]) /
                            sqrt(sigma2Total[threadIdx.y] + eps) +
                        bias[0];
                }
            } else {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[0] *
                            (input[tid +
                                   (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                             muTotal[threadIdx.y]) /
                            sqrt(sigma2Total[threadIdx.y] + eps) +
                        bias[0];
                }
            }
        }
    }
}
namespace infini {
void LaynormKernel(const float *input, const float *scale, const float *bias,
                   const float eps, int size, int scaleSize, int biasSize,
                   const int dimsize, const int stride, float *output) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockLaynormKernel<1024>
            <<<num_block, BLOCK_DIM>>>(input, scale, bias, eps, scaleSize,
                                       biasSize, dimsize, stride, output);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<32, 32><<<grid_dim, block_dim>>>(
            input, scale, bias, eps, scaleSize, biasSize, dimsize, num_block,
            stride, output);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<16, 64><<<grid_dim, block_dim>>>(
            input, scale, bias, eps, scaleSize, biasSize, dimsize, num_block,
            stride, output);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<8, 128><<<grid_dim, block_dim>>>(
            input, scale, bias, eps, scaleSize, biasSize, dimsize, num_block,
            stride, output);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<4, 256><<<grid_dim, block_dim>>>(
            input, scale, bias, eps, scaleSize, biasSize, dimsize, num_block,
            stride, output);
    }
}
} // namespace infini
