#include "core/data_type.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_utility.h"
#include <cstdio>

__global__ void cudaPrintFloatImpl(float *x, int len) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    if (start == 0) {
        for (int i = 0; i < len; ++i) {
            printf("%.7f ", x[i]);
        }
        printf("\n");
    }
}

namespace infini {

void cudaPrintFloat(float *x, int len) {
    cudaPrintFloatImpl<<<1, 1>>>(x, len);
    cudaDeviceSynchronize();
}

void cudaPrintTensor(const Tensor &tensor) {
    cudaPrintFloat(tensor->getRawDataPtr<float *>(), tensor->size());
}

cudnnDataType_t cudnnDataTypeConvert(DataType dataType) {
    if (dataType == DataType::Float32) {
        return CUDNN_DATA_FLOAT;
    }
    if (dataType == DataType::Double) {
        return CUDNN_DATA_DOUBLE;
    }
    if (dataType == DataType::Float16) {
        return CUDNN_DATA_HALF;
    }
    if (dataType == DataType::Int8) {
        return CUDNN_DATA_INT8;
    }
    if (dataType == DataType::Int32) {
        return CUDNN_DATA_INT32;
    }
    if (dataType == DataType::UInt8) {
        return CUDNN_DATA_UINT8;
    }
    if (dataType == DataType::BFloat16) {
        return CUDNN_DATA_BFLOAT16;
    }
    if (dataType == DataType::Int64) {
        return CUDNN_DATA_INT64;
    }
    if (dataType == DataType::Bool) {
        return CUDNN_DATA_BOOLEAN;
    }
    IT_ASSERT(false, "Unsupported data type");
}

cudaDataType cublasDataTypeConvert(DataType dataType) {
    switch (dataType.getIndex()) {
    case 1:
        return CUDA_R_32F;
    // case 3:
    //     return CUDA_R_8I;
    case 10:
        return CUDA_R_16F;
    case 11:
        return CUDA_R_64F;
    // case 16:
    //     return CUDA_R_16BF;
    default:
        IT_ASSERT(false, "MatMul Unsupported data type");
    }
}
} // namespace infini
