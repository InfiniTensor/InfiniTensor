#include "cuda/cuda_common.h"
#include <cub/cub.cuh>

template <typename T, int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void blockDequantizeLinearKernel(const uint8_t *inputX, const T *inputScale,
                                     T *output, const int dimsize,
                                     const int stride,
                                     const uint8_t *inputZeroPoint) {
    // len(scale) = len(bias) = dimsize
    int tmp = blockIdx.x % stride;
    int tid = (blockIdx.x - tmp) * dimsize + tmp;
    int remain = dimsize % BLOCK_DIM;
    int step = (dimsize - remain) / BLOCK_DIM + 1;
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            output[tid + (threadIdx.x * step + ind) * stride] =
                static_cast<T>(
                    inputX[tid + (threadIdx.x * step + ind) * stride] -
                    inputZeroPoint[threadIdx.x * step + ind]) *
                inputScale[threadIdx.x * step + ind];
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            output[tid +
                   (remain * step + (threadIdx.x - remain) * (step - 1) + ind) *
                       stride] =
                static_cast<T>(
                    inputX[tid + (remain * step +
                                  (threadIdx.x - remain) * (step - 1) + ind) *
                                     stride] -
                    inputZeroPoint[remain * step +
                                   (threadIdx.x - remain) * (step - 1) + ind]) *
                inputScale[remain * step + (threadIdx.x - remain) * (step - 1) +
                           ind];
        }
    }
}
template <typename T, int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void blockDequantizeLinearKernel(const uint8_t *inputX, const T *inputScale,
                                     T *output, const int dimsize,
                                     const int stride) {
    // len(scale) = len(bias) = dimsize
    int tmp = blockIdx.x % stride;
    int tid = (blockIdx.x - tmp) * dimsize + tmp;
    int remain = dimsize % BLOCK_DIM;
    int step = (dimsize - remain) / BLOCK_DIM + 1;
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            output[tid + (threadIdx.x * step + ind) * stride] =
                static_cast<T>(
                    inputX[tid + (threadIdx.x * step + ind) * stride]) *
                inputScale[threadIdx.x * step + ind];
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            output[tid +
                   (remain * step + (threadIdx.x - remain) * (step - 1) + ind) *
                       stride] =
                static_cast<T>(
                    inputX[tid + (remain * step +
                                  (threadIdx.x - remain) * (step - 1) + ind) *
                                     stride]) *
                inputScale[remain * step + (threadIdx.x - remain) * (step - 1) +
                           ind];
        }
    }
}

template <typename T, int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void
warpDequantizeLinearKernel(const uint8_t *inputX, const T *inputScale,
                           T *output, const int dimsize, const int otherSize,
                           const int stride, const uint8_t *inputZeroPoint) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    int remain = dimsize % BLOCK_DIM_x;
    int step = (dimsize - remain) / BLOCK_DIM_x + 1;
    if (otherIdx < otherSize) {
        if (threadIdx.x < remain) {
            for (int ind = 0; ind < step; ind++) {
                output[tid + (threadIdx.x * step + ind) * stride] =
                    static_cast<T>(
                        inputX[tid + (threadIdx.x * step + ind) * stride] -
                        inputZeroPoint[threadIdx.x * step + ind]) *
                    inputScale[threadIdx.x * step + ind];
            }
        } else {
            for (int ind = 0; ind < step - 1; ind++) {
                output[tid + (remain * step +
                              (threadIdx.x - remain) * (step - 1) + ind) *
                                 stride] =
                    static_cast<T>(
                        inputX[tid +
                               (remain * step +
                                (threadIdx.x - remain) * (step - 1) + ind) *
                                   stride] -
                        inputZeroPoint[remain * step +
                                       (threadIdx.x - remain) * (step - 1) +
                                       ind]) *
                    inputScale[remain * step +
                               (threadIdx.x - remain) * (step - 1) + ind];
            }
        }
    }
}
template <typename T, int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void
warpDequantizeLinearKernel(const uint8_t *inputX, const T *inputScale,
                           T *output, const int dimsize, const int otherSize,
                           const int stride) {
    int otherIdx = blockIdx.x * blockDim.y + threadIdx.y;

    int tid = otherIdx % stride + (otherIdx - otherIdx % stride) * dimsize;
    int remain = dimsize % BLOCK_DIM_x;
    int step = (dimsize - remain) / BLOCK_DIM_x + 1;
    if (otherIdx < otherSize) {
        if (threadIdx.x < remain) {
            for (int ind = 0; ind < step; ind++) {
                output[tid + (threadIdx.x * step + ind) * stride] =
                    static_cast<T>(
                        inputX[tid + (threadIdx.x * step + ind) * stride]) *
                    inputScale[threadIdx.x * step + ind];
            }
        } else {
            for (int ind = 0; ind < step - 1; ind++) {
                output[tid + (remain * step +
                              (threadIdx.x - remain) * (step - 1) + ind) *
                                 stride] =
                    static_cast<T>(
                        inputX[tid +
                               (remain * step +
                                (threadIdx.x - remain) * (step - 1) + ind) *
                                   stride]) *
                    inputScale[remain * step +
                               (threadIdx.x - remain) * (step - 1) + ind];
            }
        }
    }
}
namespace infini {
void DequantizeLinearKernel(const uint8_t *inputX, const float *inputScale,
                            float *output, const int dimsize, const int stride,
                            const uint8_t *inputZeroPoint, const int size) {

    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockDequantizeLinearKernel<float, 1024><<<num_block, BLOCK_DIM, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, stride, inputZeroPoint);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 32, 32>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 16, 64>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 8, 128>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 4, 256>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    }
}

void DequantizeLinearKernel(const uint8_t *inputX, const float *inputScale,
                            float *output, const int dimsize, const int stride,
                            const int size) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockDequantizeLinearKernel<float, 1024><<<num_block, BLOCK_DIM, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, stride);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 32, 32><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 16, 64><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 8, 128><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<float, 4, 256><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    }
}
//-------------
void DequantizeLinearKernel(const uint8_t *inputX, const half *inputScale,
                            half *output, const int dimsize, const int stride,
                            const uint8_t *inputZeroPoint, const int size) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockDequantizeLinearKernel<half, 1024><<<num_block, BLOCK_DIM, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, stride, inputZeroPoint);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 32, 32>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 16, 64>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 8, 128>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 4, 256>
            <<<grid_dim, block_dim, 0, CUDAStream::stream>>>(inputX, inputScale, output, dimsize,
                                      num_block, stride, inputZeroPoint);
    }
}

void DequantizeLinearKernel(const uint8_t *inputX, const half *inputScale,
                            half *output, const int dimsize, const int stride,
                            const int size) {
    int num_block = size / dimsize;
    if (dimsize > 1024) {
        int BLOCK_DIM = 1024;

        blockDequantizeLinearKernel<half, 1024><<<num_block, BLOCK_DIM, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, stride);
    } else if (dimsize > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 32, 32><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    } else if (dimsize > 15) {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 16, 64><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    } else if (dimsize > 7) {
        int BLOCK_DIM_x = 8;
        int BLOCK_DIM_y = 128;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 8, 128><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    } else {
        int BLOCK_DIM_x = 4;
        int BLOCK_DIM_y = 256;
        int num_block_x = (num_block + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);

        warpDequantizeLinearKernel<half, 4, 256><<<grid_dim, block_dim, 0, CUDAStream::stream>>>(
            inputX, inputScale, output, dimsize, num_block, stride);
    }
}

} // namespace infini
