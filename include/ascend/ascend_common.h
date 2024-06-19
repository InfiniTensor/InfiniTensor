#pragma once
#include "acl/acl.h"
#include "acl/acl_op.h"
#include "core/common.h"
#include "core/data_type.h"

#define checkASCENDError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (ACL_SUCCESS != err) {                                              \
            fprintf(stderr, "ASCEND error in %s:%i : .\n", __FILE__,           \
                    __LINE__);                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

using ASCENDPtr = void *;

inline aclDataType aclnnDataTypeConvert(DataType dataType) {
    if (dataType == DataType::Float32) {
        return ACL_FLOAT;
    }
    if (dataType == DataType::Float16) {
        return ACL_FLOAT16;
    }
    if (dataType == DataType::Double) {
        return ACL_DOUBLE;
    }
    if (dataType == DataType::Int8) {
        return ACL_INT8;
    }
    if (dataType == DataType::Int32) {
        return ACL_INT32;
    }
    if (dataType == DataType::UInt8) {
        return ACL_UINT8;
    }
    if (dataType == DataType::BFloat16) {
        return ACL_BF16;
    }
    if (dataType == DataType::Int64) {
        return ACL_INT64;
    }
    if (dataType == DataType::Bool) {
        return ACL_BOOL;
    }
    IT_TODO_HALT_MSG("Data type " + dataType.toString() +
                     " not supported in CNNL.");
}
} // namespace infini
