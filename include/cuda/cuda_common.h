#pragma once
#include "core/common.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cudnn.h>
#include <curand.h>
#include <memory>

#define checkCudaError(call)                                                   \
    if (auto err = call; err != cudaSuccess)                                   \
    throw ::infini::Exception(std::string("[") + __FILE__ + ":" +              \
                              std::to_string(__LINE__) + "] CUDA error (" +    \
                              #call + "): " + cudaGetErrorString(err))

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
    if (auto err = call; err != CUDNN_STATUS_SUCCESS)                          \
    throw ::infini::Exception(std::string("[") + __FILE__ + ":" +              \
                              std::to_string(__LINE__) + "] cuDNN error (" +   \
                              #call + "): " + cudnnGetErrorString(err))

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

class CUDAStream {
  public:
    CUDAStream(const CUDAStream &) = delete;
    CUDAStream(CUDAStream &&) = delete;
    void operator=(const CUDAStream &) = delete;
    void operator=(CUDAStream &&) = delete;
    static cudaStream_t getCurrentStream() { return _stream; }
    static void Init() { CUDAStream::_stream = 0; };
    static void createStream() { checkCudaError(cudaStreamCreate(&_stream)); }
    static void destroyStream() { checkCudaError(cudaStreamDestroy(_stream)); }

  private:
    CUDAStream() {};
    static cudaStream_t _stream;
};

} // namespace infini
