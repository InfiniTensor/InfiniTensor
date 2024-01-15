#include "cuda/cuda_common.h"
#include <cub/cub.cuh>

template <typename T, int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void blockLaynormKernel(const T *input, const T *scale, const int dimsize,
                            const int stride, T *output, const T eps,
                            int scaleSize, const T *bias, int biasSize) {
    // len(scale) = len(bias) = dimsize
    int tmp = blockIdx.x % stride;
    int tid = (blockIdx.x - tmp) * dimsize + tmp;
    T muPartial = 0.0f;
    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
        muPartial += input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride];
    }
    typedef cub::BlockReduce<T, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;
    T muBlock = BlockReduce(temp_storage).Reduce(muPartial, cub::Sum());
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        mu = muBlock * static_cast<T>(__fdividef(1.0F, dimsize));
    }
    __syncthreads();

    T sigma2Partial = 0.0f;
    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
        sigma2Partial +=
            (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu) *
            (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu);
    }
    typedef cub::BlockReduce<T, BLOCK_DIM> BlockReduce;

    __shared__ T sigma2;
    T sigma2Block = BlockReduce(temp_storage).Reduce(sigma2Partial, cub::Sum());
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        sigma2 = sigma2Block * static_cast<T>(__fdividef(1.0F, dimsize));
    }
    __syncthreads();
    if (biasSize == dimsize) {
        if (scaleSize == dimsize) {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[threadIdx.x + ph * BLOCK_DIM] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) *
                        static_cast<T>(__fdividef(
                            1.0F, sqrt(static_cast<float>(sigma2 + eps)))) +
                    bias[threadIdx.x + ph * BLOCK_DIM];
            }
        } else {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[0] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) *
                        static_cast<T>(__fdividef(
                            1.0F, sqrt(static_cast<float>(sigma2 + eps)))) +
                    bias[threadIdx.x + ph * BLOCK_DIM];
            }
        }
    } else {
        if (scaleSize == dimsize) {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[threadIdx.x + ph * BLOCK_DIM] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) *
                        static_cast<T>(__fdividef(
                            1.0F, sqrt(static_cast<float>(sigma2 + eps)))) +
                    bias[0];
            }
        } else {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                    scale[0] *
                        (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] -
                         mu) *
                        static_cast<T>(__fdividef(
                            1.0F, sqrt(static_cast<float>(sigma2 + eps)))) +
                    bias[0];
            }
        }
    }
}
//-----------------
template <typename T, int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void blockLaynormKernel(const T *input, const T *scale, const int dimsize,
                            const int stride, T *output, const T eps,
                            int scaleSize) {
    // len(scale) = len(bias) = dimsize
    int tmp = blockIdx.x % stride;
    int tid = (blockIdx.x - tmp) * dimsize + tmp;
    T muPartial = 0.0f;
    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
        muPartial += input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride];
    }
    typedef cub::BlockReduce<T, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;
    T muBlock = BlockReduce(temp_storage).Reduce(muPartial, cub::Sum());
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        mu = muBlock * static_cast<T>(__fdividef(1.0F, dimsize));
    }
    __syncthreads();

    T sigma2Partial = 0.0f;
    for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {
        sigma2Partial +=
            (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu) *
            (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu);
    }
    typedef cub::BlockReduce<T, BLOCK_DIM> BlockReduce;

    __shared__ T sigma2;
    T sigma2Block = BlockReduce(temp_storage).Reduce(sigma2Partial, cub::Sum());
    if (threadIdx.x ==
        0) { // must set threadIdx.x = 0 write the output to memory
        sigma2 = sigma2Block * static_cast<T>(__fdividef(1.0F, dimsize));
    }
    __syncthreads();
    if (scaleSize == dimsize) {
        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

            output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                scale[threadIdx.x + ph * BLOCK_DIM] *
                (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu) *
                static_cast<T>(
                    __fdividef(1.0F, sqrt(static_cast<float>(sigma2 + eps))));
        }
    } else {
        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM < dimsize; ph++) {

            output[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] =
                scale[0] *
                (input[tid + (threadIdx.x + ph * BLOCK_DIM) * stride] - mu) *
                static_cast<T>(
                    __fdividef(1.0F, sqrt(static_cast<float>(sigma2 + eps))));
        }
    }
}
//-----------------
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
template <typename T, int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void warpLaynormKernel(const T *input, const T *scale,
                                  const int dimsize, const int stride,
                                  T *output, const T eps, int scaleSize,
                                  int otherSize, const T *bias, int biasSize) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    if (otherIdx < otherSize) {

        __shared__ T muTotal[BLOCK_DIM_y];
        __shared__ T sigma2Total[BLOCK_DIM_y];

        T muPartial = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            muPartial += input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride];
        }

        muPartial = WarpAllReduce<SumOp, T, BLOCK_DIM_x>(muPartial);

        if (threadIdx.x == 0)
            muTotal[threadIdx.y] =
                muPartial * static_cast<T>(__fdividef(1.0F, dimsize));

        //--------------------------------------------
        T sigma2Partial = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            sigma2Partial +=
                (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                 muTotal[threadIdx.y]) *
                (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                 muTotal[threadIdx.y]);
        }

        sigma2Partial = WarpAllReduce<SumOp, T, BLOCK_DIM_x>(sigma2Partial);

        if (threadIdx.x == 0)
            sigma2Total[threadIdx.y] =
                sigma2Partial * static_cast<T>(__fdividef(1.0F, dimsize));

        //--------------------------------------------
        if (biasSize == dimsize) {
            if (scaleSize == dimsize) {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[threadIdx.x + ph * BLOCK_DIM_x] *
                            (input[tid +
                                   (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                             muTotal[threadIdx.y]) *
                            static_cast<T>(__fdividef(
                                1.0F, sqrt(static_cast<float>(
                                          sigma2Total[threadIdx.y] + eps)))) +
                        bias[threadIdx.x + ph * BLOCK_DIM_x];
                }
            } else {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[0] *
                            (input[tid +
                                   (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                             muTotal[threadIdx.y]) *
                            static_cast<T>(__fdividef(
                                1.0F, sqrt(static_cast<float>(
                                          sigma2Total[threadIdx.y] + eps)))) +
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
                             muTotal[threadIdx.y]) *
                            static_cast<T>(__fdividef(
                                1.0F, sqrt(static_cast<float>(
                                          sigma2Total[threadIdx.y] + eps)))) +
                        bias[0];
                }
            } else {
                for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize;
                     ph++) {

                    output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                        scale[0] *
                            (input[tid +
                                   (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                             muTotal[threadIdx.y]) *
                            static_cast<T>(__fdividef(
                                1.0F, sqrt(static_cast<float>(
                                          sigma2Total[threadIdx.y] + eps)))) +
                        bias[0];
                }
            }
        }
    }
}
template <typename T, int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void warpLaynormKernel(const T *input, const T *scale,
                                  const int dimsize, const int stride,
                                  T *output, const T eps, int scaleSize,
                                  int otherSize) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    if (otherIdx < otherSize) {

        __shared__ T muTotal[BLOCK_DIM_y];
        __shared__ T sigma2Total[BLOCK_DIM_y];

        T muPartial = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            muPartial += input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride];
        }

        muPartial = WarpAllReduce<SumOp, T, BLOCK_DIM_x>(muPartial);

        if (threadIdx.x == 0)
            muTotal[threadIdx.y] =
                muPartial * static_cast<T>(__fdividef(1.0F, dimsize));

        //--------------------------------------------
        T sigma2Partial = 0.0f;

        for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {
            sigma2Partial +=
                (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                 muTotal[threadIdx.y]) *
                (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                 muTotal[threadIdx.y]);
        }

        sigma2Partial = WarpAllReduce<SumOp, T, BLOCK_DIM_x>(sigma2Partial);

        if (threadIdx.x == 0)
            sigma2Total[threadIdx.y] =
                sigma2Partial * static_cast<T>(__fdividef(1.0F, dimsize));

        //--------------------------------------------
        if (scaleSize == dimsize) {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                    scale[threadIdx.x + ph * BLOCK_DIM_x] *
                    (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                     muTotal[threadIdx.y]) *
                    static_cast<T>(
                        __fdividef(1.0F, sqrt(static_cast<float>(
                                             sigma2Total[threadIdx.y] + eps))));
            }
        } else {
            for (int ph = 0; threadIdx.x + ph * BLOCK_DIM_x < dimsize; ph++) {

                output[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] =
                    scale[0] *
                    (input[tid + (threadIdx.x + ph * BLOCK_DIM_x) * stride] -
                     muTotal[threadIdx.y]) *
                    static_cast<T>(
                        __fdividef(1.0F, sqrt(static_cast<float>(
                                             sigma2Total[threadIdx.y] + eps))));
            }
        }
    }
}
namespace infini {
void LaynormKernel(const float *input, const float *scale, const float eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   float *output, const float *bias, int biasSize) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockLaynormKernel<float, 1024>
            <<<num_block, BLOCK_DIM>>>(input, scale, dimsize, stride, output,
                                       eps, scaleSize, bias, biasSize);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 32, 32><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 16, 64><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 8, 128><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 4, 256><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    }
}

void LaynormKernel(const float *input, const float *scale, const float eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   float *output) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockLaynormKernel<float, 1024><<<num_block, BLOCK_DIM>>>(
            input, scale, dimsize, stride, output, eps, scaleSize);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 32, 32><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 16, 64><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 8, 128><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<float, 4, 256><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    }
}
//-----------------
void LaynormKernel(const half *input, const half *scale, const half eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   half *output, const half *bias, int biasSize) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockLaynormKernel<half, 1024>
            <<<num_block, BLOCK_DIM>>>(input, scale, dimsize, stride, output,
                                       eps, scaleSize, bias, biasSize);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 32, 32><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 16, 64><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 8, 128><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 4, 256><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block,
            bias, biasSize);
    }
}

void LaynormKernel(const half *input, const half *scale, const half eps,
                   int size, int scaleSize, const int dimsize, const int stride,
                   half *output) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockLaynormKernel<half, 1024><<<num_block, BLOCK_DIM>>>(
            input, scale, dimsize, stride, output, eps, scaleSize);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 32, 32><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 16, 64><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 8, 128><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpLaynormKernel<half, 4, 256><<<grid_dim, block_dim>>>(
            input, scale, dimsize, stride, output, eps, scaleSize, num_block);
    }
}
} // namespace infini
