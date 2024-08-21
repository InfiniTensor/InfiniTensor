#ifndef COMMON_H
#define COMMON_H
#pragma once

#include "omp.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <curand.h>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <pybind11/pybind11.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace tpm {

#define checkCudaError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (cudaSuccess != err) {                                              \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

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

#define checkCublasError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (CUBLAS_STATUS_SUCCESS != err) {                                    \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cublasGetErrorString(err));            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define checkCudnnError(call)                                                  \
    {                                                                          \
        auto err = call;                                                       \
        if (CUDNN_STATUS_SUCCESS != err) {                                     \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudnnGetErrorString(err));             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
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

#define checkCurandError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (CURAND_STATUS_SUCCESS != err) {                                    \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, curandGetErrorString(err));            \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

class Tensor;
class Operator;
class Graph;
class SubGraph;

using TensorVec = std::vector<Tensor *>;
using TensorMap = std::map<size_t, Tensor *>;
using OpVec = std::vector<Operator *>;
using OpMap = std::map<size_t, Operator *>;
using VType = uint32_t;
using SplittingPoints = std::vector<std::vector<int>>;

inline size_t generateGuid() {
    static size_t guid = 0;
    return guid++;
}

inline uint64_t generateHash() {
    static uint64_t tag = 0;
    uint64_t hash = std::hash<uint64_t>()(tag++);
    return hash;
}

inline uint64_t hashAppend(uint64_t a, uint64_t b) {
    return (a * 10000019 + b * 10000079) % 2147483647;
}

inline uint64_t hashPack(uint64_t x) { return (x * 10000103) % 2147483647; }

inline VType powVType(VType val, int pow) {
    VType ret = 1;
    for (int i = 0; i < pow; ++i)
        ret *= val;
    return ret;
}

static int gcd(int a, int b) {
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

static int gcd(std::vector<int> &vec) {
    if (vec.empty())
        return 1;
    int ret = vec[0];
    for (size_t i = 1, iEnd = vec.size(); i < iEnd; ++i)
        ret = gcd(ret, vec[i]);
    return ret;
}

static int max(std::vector<int> &vec) {
    if (vec.empty())
        return 0;
    int ret = vec[0];
    for (size_t i = 1, iEnd = vec.size(); i < iEnd; ++i)
        ret = std::max(ret, vec[i]);
    return ret;
}
} // end of namespace tpm

#endif // COMMON_H
