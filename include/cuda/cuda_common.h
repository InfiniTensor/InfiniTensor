#pragma once
#include "core/common.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cudnn.h>
#include <curand.h>

// TODO: replace with Exception (IT_ASSERT)
#define checkCudaError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (cudaSuccess != err) {                                              \
            fprintf(stderr, "Cuda error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            IT_ASSERT(false);                                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define checkCUresult(call)                                                    \
    {                                                                          \
        auto err = call;                                                       \
        const char *errName;                                                   \
        if (CUDA_SUCCESS != err) {                                             \
            cuGetErrorString(err, &errName);                                   \
            IT_ASSERT(err == CUDA_SUCCESS,                                     \
                      (string("CU error: ") + string(errName)));               \
        }                                                                      \
    }

#define checkCublasError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (CUBLAS_STATUS_SUCCESS != err) {                                    \
            fprintf(stderr, "cuBLAS error in %s:%i : %s.\n", __FILE__,         \
                    __LINE__, cublasGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define checkCudnnError(call)                                                  \
    {                                                                          \
        auto err = call;                                                       \
        if (CUDNN_STATUS_SUCCESS != err) {                                     \
            fprintf(stderr, "cuDNN error in %s:%i : %s.\n", __FILE__,          \
                    __LINE__, cudnnGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define checkCurandError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (CURAND_STATUS_SUCCESS != err) {                                    \
            fprintf(stderr, "cuRAND error in %s:%i : %s.\n", __FILE__,         \
                    __LINE__, curandGetErrorString(err));                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

inline const char *cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

inline const char *curandGetErrorString(curandStatus_t error) {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

using CudaPtr = void *;

} // namespace infini
