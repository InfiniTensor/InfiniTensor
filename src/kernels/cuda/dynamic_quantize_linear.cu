#include <cub/cub.cuh>

#include "cuda/cuda_common.h"

__device__ float _saturate(float x) {
    return x < 0.f ? 0.f : (x > 255.0 ? 255.0 : x);
}

template <class T>
__device__ __forceinline__ static T max___(T a, T b) noexcept {
    return a > b ? a : b;
}

template <class T>
__device__ __forceinline__ static T min___(T a, T b) noexcept {
    return a < b ? a : b;
}

template <int BLOCK_DIM>
__launch_bounds__(BLOCK_DIM) __global__
    void _dynamicQuantizeLinearKernel(float *input, uint8_t *outputY,
                                      float *yScale, uint8_t *yZeroPoint,
                                      int size) {
    int i = threadIdx.x + blockIdx.x * BLOCK_DIM;
    float maxData = __FLT_MAX__;
    float minData = -__FLT_MAX__;
    int remain = size % BLOCK_DIM;
    int step = (size - remain) / BLOCK_DIM + 1;

    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            maxData = max___(maxData, input[threadIdx.x * step + ind]);
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            maxData = max___(maxData,
                             input[remain * step +
                                   (threadIdx.x - remain) * (step - 1) + ind]);
        }
    }
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            minData = min___(minData, input[threadIdx.x * step + ind]);
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            minData = min___(minData,
                             input[remain * step +
                                   (threadIdx.x - remain) * (step - 1) + ind]);
        }
    }
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float maxTotal;
    float blockMax = BlockReduce(temp_storage).Reduce(maxData, cub::Max());

    __shared__ float minTotal;
    float blockMin = BlockReduce(temp_storage).Reduce(minData, cub::Min());
    if (threadIdx.x == 0) {
        maxTotal = blockMax;
        minTotal = blockMin;
    }
    __syncthreads();
    int qmax = 255;
    int qmin = 0;
    float absMax = max___(abs(maxTotal), abs(minTotal));
    yScale[0] = absMax * 2 / (254 - qmin);
    float intermediate_zero_point = 254 - absMax / yScale[0];
    float _yZeroPoint = round(_saturate(intermediate_zero_point));
    yZeroPoint[0] = static_cast<uint8_t>(_yZeroPoint);
    if (i < size) {
        outputY[i] = static_cast<uint8_t>(
            _saturate(round(input[i] / yScale[0]) + _yZeroPoint));
    }
}
//----------

template <int BLOCK_DIM, int numPerThread>
__launch_bounds__(BLOCK_DIM) __global__
    void _dynamicQuantizeLinearKernel(float *input, uint8_t *outputY,
                                      float *yScale, uint8_t *yZeroPoint,
                                      int size) {
    int i = threadIdx.x + blockIdx.x * BLOCK_DIM;
    float maxData = __FLT_MAX__;
    float minData = -__FLT_MAX__;
    int remain = size % BLOCK_DIM;
    int step = (size - remain) / BLOCK_DIM + 1;
    float dataPerThread[numPerThread];
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            dataPerThread[ind] = input[threadIdx.x * step + ind];
            maxData = max___(maxData, dataPerThread[ind]);
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            dataPerThread[ind] =
                input[remain * step + (threadIdx.x - remain) * (step - 1) +
                      ind];
            maxData = max___(maxData, dataPerThread[ind]);
        }
    }
    if (threadIdx.x < remain) {
        for (int ind = 0; ind < step; ind++) {
            minData = min___(minData, dataPerThread[ind]);
        }
    } else {
        for (int ind = 0; ind < step - 1; ind++) {
            minData = min___(minData, dataPerThread[ind]);
        }
    }
    typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float maxTotal;
    float blockMax = BlockReduce(temp_storage).Reduce(maxData, cub::Max());

    __shared__ float minTotal;
    float blockMin = BlockReduce(temp_storage).Reduce(minData, cub::Min());
    if (threadIdx.x == 0) {
        maxTotal = blockMax;
        minTotal = blockMin;
    }
    __syncthreads();
    int qmax = 255;
    int qmin = 0;
    float absMax = max___(abs(maxTotal), abs(minTotal));
    yScale[0] = absMax * 2 / (254 - qmin);
    float intermediate_zero_point = 254 - absMax / yScale[0];
    float _yZeroPoint = round(_saturate(intermediate_zero_point));
    yZeroPoint[0] = static_cast<uint8_t>(_yZeroPoint);
    if (i < size) {
        outputY[i] = static_cast<uint8_t>(
            _saturate(round(input[i] / yScale[0]) + _yZeroPoint));
    }
}

namespace infini {
void dynamicQuantizeLinearKernel(float *input, uint8_t *outputY, float *yScale,
                                 uint8_t *yZeroPoint, int size) {
    if (size > 1024 * 128) {
        int BLOCK_DIM = 1024;
        int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
        _dynamicQuantizeLinearKernel<1024><<<num_blocks, BLOCK_DIM>>>(
            input, outputY, yScale, yZeroPoint, size);
    } else if (size > 1024 * 64) {
        int BLOCK_DIM = 1024;
        int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
        _dynamicQuantizeLinearKernel<1024, 128><<<num_blocks, BLOCK_DIM>>>(
            input, outputY, yScale, yZeroPoint, size);
    } else if (size > 1024 * 32) {
        int BLOCK_DIM = 1024;
        int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
        _dynamicQuantizeLinearKernel<1024, 64><<<num_blocks, BLOCK_DIM>>>(
            input, outputY, yScale, yZeroPoint, size);
    } else if (size > 1024 * 16) {
        int BLOCK_DIM = 1024;
        int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
        _dynamicQuantizeLinearKernel<1024, 32><<<num_blocks, BLOCK_DIM>>>(
            input, outputY, yScale, yZeroPoint, size);
    } else if (size > 1024 * 4) {
        int BLOCK_DIM = 1024;
        int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
        _dynamicQuantizeLinearKernel<1024, 16><<<num_blocks, BLOCK_DIM>>>(
            input, outputY, yScale, yZeroPoint, size);
    } else if (size > 1024) {
        int BLOCK_DIM = 1024;
        int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
        _dynamicQuantizeLinearKernel<1024, 4><<<num_blocks, BLOCK_DIM>>>(
            input, outputY, yScale, yZeroPoint, size);
    } else {
        int BLOCK_DIM = 1024;
        int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
        _dynamicQuantizeLinearKernel<1024, 1><<<num_blocks, BLOCK_DIM>>>(
            input, outputY, yScale, yZeroPoint, size);
    }
}
} // namespace infini
