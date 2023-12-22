#include "cuda/cuda_common.h"

constexpr unsigned int num_threads() { return 32 * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }

__global__ void _subA_kernel(void *a, void *b, int size, int k, int delta) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        int j = delta * (i - i % k) / k;
        ((int8_t *)a)[i] = ((int8_t *)a)[i] - ((int8_t *)b)[j];
    }
}

__global__ void _subA_u8_kernel(void *a, void *b, int size, int k, int delta) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        int j = delta * (i - i % k) / k;
        auto aData = static_cast<int16_t>(((uint8_t *)a)[i]);
        auto bData = static_cast<int16_t>(((uint8_t *)b)[j]);
        ((int8_t *)a)[i] = static_cast<int8_t>(aData - bData);
    }
}

__global__ void _subB_kernel(void *a, void *b, int size, int k, int n,
                             int delta) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        int j = delta * (i / k) + (i % n);
        ((int8_t *)a)[i] = ((int8_t *)a)[i] - ((int8_t *)b)[j];
    }
}

__global__ void _subB_u8_kernel(void *a, void *b, int size, int k, int n,
                                int delta) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        int j = delta * (i / k) + (i % n);
        auto aData = static_cast<int16_t>(((uint8_t *)a)[i]);
        auto bData = static_cast<int16_t>(((uint8_t *)b)[j]);
        ((int8_t *)a)[i] = static_cast<int8_t>(aData - bData);
    }
}

namespace infini {
void subA_kernel(int dType, void *a, void *b, int size, int k, int delta) {

    int blocksize = block_work_size();
    int gridsize = (size + block_work_size() - 1) / block_work_size();
    if (dType == 3) {
        _subA_kernel<<<gridsize, blocksize, 0, CUDAStream::stream>>>(a, b, size, k, delta);
    } else if (dType == 2) {
        _subA_u8_kernel<<<gridsize, blocksize, 0, CUDAStream::stream>>>(a, b, size, k, delta);
    } else {
        IT_TODO_HALT();
    }
}
void subB_kernel(int dType, void *a, void *b, int size, int k, int n,
                 int delta) {

    int blocksize = block_work_size();
    int gridsize = (size + block_work_size() - 1) / block_work_size();
    if (dType == 3) {
        _subB_kernel<<<gridsize, blocksize, 0, CUDAStream::stream>>>(a, b, size, k, n, delta);
    } else if (dType == 2) {
        _subB_u8_kernel<<<gridsize, blocksize, 0, CUDAStream::stream>>>(a, b, size, k, n, delta);
    } else {
        IT_TODO_HALT();
    }
}
}; // namespace infini