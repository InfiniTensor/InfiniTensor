#pragma once
#include "cnnl.h"
#include "cnrt.h"
#include "core/common.h"
#include "core/data_type.h"

#define checkBangError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (CNRT_RET_SUCCESS != err) {                                         \
            fprintf(stderr, "Bang error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    cnrtGetErrorStr(err));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define checkCnnlError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (CNNL_STATUS_SUCCESS != err) {                                      \
            fprintf(stderr, "cnnl error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    cnnlGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

using BangPtr = void *;

inline cnnlDataType_t cnnlDataTypeConvert(DataType dataType) {
    if (dataType == DataType::Float32) {
        return CNNL_DTYPE_FLOAT;
    }
    if (dataType == DataType::Float16) {
        return CNNL_DTYPE_HALF;
    }
    if (dataType == DataType::Double) {
        return CNNL_DTYPE_DOUBLE;
    }
    if (dataType == DataType::Int8) {
        return CNNL_DTYPE_INT8;
    }
    if (dataType == DataType::Int32) {
        return CNNL_DTYPE_INT32;
    }
    if (dataType == DataType::UInt8) {
        return CNNL_DTYPE_UINT8;
    }
    if (dataType == DataType::BFloat16) {
        return CNNL_DTYPE_BFLOAT16;
    }
    if (dataType == DataType::Int64) {
        return CNNL_DTYPE_INT64;
    }
    if (dataType == DataType::Bool) {
        return CNNL_DTYPE_BOOL;
    }
    return CNNL_DTYPE_INVALID;
}

} // namespace infini
