// #include "cuda/cuda_common.h"
// #include "cuda/cuda_argmax.h"



// namespace infini {
//     void argmax_kernel(void * input, int64_t *output, const int * inputShape, int ndim,
//                       int axis, bool keepdims, bool selectLastIndex) {
//         for(int i = 0; i < ndim; i++) {
//             std::cout << "inputShape[" << i << "] is " << inputShape[i] << std::endl;
//         }
//         std::cout << "ndim is " << ndim << std::endl;
//         std::cout << "axis is " << axis << std::endl;
//         std::cout << "keepdims is " << keepdims << std::endl;
//         std::cout << "selectLastIndex is " << selectLastIndex << std::endl;
//     }

// } // namespace infini

#include "cuda/cuda_common.h"
#include "cuda/cuda_argmax.h"
#include <iostream>
#include <cuda_runtime.h>
#include <limits>

namespace infini {

template <typename T>
__global__ void argmax_kernel_impl(const T* input, int64_t* output, 
                                  const int* inputShape, int ndim,
                                  int axis, bool keepdims, 
                                  bool selectLastIndex) {
    // 计算总元素数和轴相关参数
    int totalElements = 1;
    for (int i = 0; i < ndim; ++i) {
        totalElements *= inputShape[i];
    }
    
    int axisSize = inputShape[axis];
    int outerDims = totalElements / axisSize;
    
    // 每个线程块处理一个或多个外部维度
    for (int outerIndex = blockIdx.x; outerIndex < outerDims; outerIndex += gridDim.x) {
        __shared__ int sharedMaxIndex[256];
        __shared__ T sharedMaxValue[256];
        
        int tid = threadIdx.x;
        const T* segmentStart = input + outerIndex * axisSize;
        
        // 初始化共享内存
        if (tid < axisSize) {
            sharedMaxIndex[tid] = tid;
            sharedMaxValue[tid] = segmentStart[tid];
        } 
        __syncthreads();
        
        // 块内归约查找最大值索引
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < axisSize) {
                if (sharedMaxValue[tid + s] > sharedMaxValue[tid] || 
                    (sharedMaxValue[tid + s] == sharedMaxValue[tid] && 
                     selectLastIndex && (tid + s) > tid)) {
                    sharedMaxValue[tid] = sharedMaxValue[tid + s];
                    sharedMaxIndex[tid] = sharedMaxIndex[tid + s];
                }
            }
            __syncthreads();
        }
        
        // 第一个线程写入结果
        if (tid == 0) {
            output[outerIndex] = sharedMaxIndex[0];
            printf("output[%d] = %lld\n", outerIndex, output[outerIndex]);
        }
    }
}

template <typename T>
void launch_argmax_kernel(const T* input, int64_t* output, 
                         const int* inputShape, int ndim,
                         int axis, bool keepdims, 
                         bool selectLastIndex) {
    // 计算网格和块大小
    int totalElements = 1;
    for (int i = 0; i < ndim; ++i) {
        totalElements *= inputShape[i];
    }
    int outerDims = ndim - 1;
    
    dim3 blocks(std::min(outerDims, 256));
    dim3 threads(256);
    
    // 启动内核
    argmax_kernel_impl<T><<<blocks, threads>>>(input, output, inputShape, ndim, 
                                              axis, keepdims, selectLastIndex);
    
    cudaDeviceSynchronize();
}

void argmax_kernel(void* input, int64_t* output, const int* inputShape, int ndim,
                  int axis, bool keepdims, bool selectLastIndex, DataType dtype) {
    // 调试信息
    for(int i = 0; i < ndim; i++) {
        std::cout << "inputShape[" << i << "] is " << inputShape[i] << std::endl;
    }
    std::cout << "ndim is " << ndim << std::endl;
    std::cout << "axis is " << axis << std::endl;
    std::cout << "keepdims is " << keepdims << std::endl;
    std::cout << "selectLastIndex is " << selectLastIndex << std::endl;

    if (dtype == DataType::Float32) {
        std::cout << "Launching Float32 ArgMax" << std::endl;
        launch_argmax_kernel<float>(static_cast<const float*>(input), 
                               output, inputShape, ndim, 
                               axis, keepdims, selectLastIndex);

    }
    if (dtype == DataType::Float16) {
        std::cout << "Launching Float16 ArgMax" << std::endl;
        launch_argmax_kernel<half>(static_cast<const half*>(input), 
                               output, inputShape, ndim, 
                               axis, keepdims, selectLastIndex);

    }
    if (dtype == DataType::Double) {
        std::cout << "Launching Double ArgMax" << std::endl;
        launch_argmax_kernel<double>(static_cast<const double*>(input), 
                               output, inputShape, ndim, 
                               axis, keepdims, selectLastIndex);

    }
    if (dtype == DataType::Int8) {
        std::cout << "Launching Int8 ArgMax" << std::endl;
        launch_argmax_kernel<int8_t>(static_cast<const int8_t*>(input), 
                               output, inputShape, ndim, 
                               axis, keepdims, selectLastIndex);

    }
    if (dtype == DataType::Int32) {
        std::cout << "Launching Int32 ArgMax" << std::endl;
        launch_argmax_kernel<int32_t>(static_cast<const int32_t*>(input), 
                               output, inputShape, ndim, 
                               axis, keepdims, selectLastIndex);

    }
    if (dtype == DataType::UInt8) {
        std::cout << "Launching UInt8 ArgMax" << std::endl;
        launch_argmax_kernel<uint8_t>(static_cast<const uint8_t*>(input), 
                               output, inputShape, ndim, 
                               axis, keepdims, selectLastIndex);

    }
    if (dtype == DataType::Int64) {
        std::cout << "Launching Int64 ArgMax" << std::endl;
        launch_argmax_kernel<int64_t>(static_cast<const int64_t*>(input), 
                               output, inputShape, ndim, 
                               axis, keepdims, selectLastIndex);

    }

}

} // namespace infini