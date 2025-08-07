#include "utils/infinicore_utils.h"

namespace infini {

#define CASE_DTYPE_CONVERT(src, dst)                                           \
    if (dtype == DataType::src)                                                \
        return dst;

infiniDtype_t toInfiniDtype(const DataType &dtype) {
    CASE_DTYPE_CONVERT(Undefine, INFINI_DTYPE_INVALID)
    CASE_DTYPE_CONVERT(Float32, INFINI_DTYPE_F32)
    CASE_DTYPE_CONVERT(Float16, INFINI_DTYPE_F16)
    CASE_DTYPE_CONVERT(Double, INFINI_DTYPE_F64)
    CASE_DTYPE_CONVERT(BFloat16, INFINI_DTYPE_BF16)
    CASE_DTYPE_CONVERT(Int8, INFINI_DTYPE_I8)
    CASE_DTYPE_CONVERT(Int16, INFINI_DTYPE_I16)
    CASE_DTYPE_CONVERT(Int32, INFINI_DTYPE_I32)
    CASE_DTYPE_CONVERT(Int64, INFINI_DTYPE_I64)
    CASE_DTYPE_CONVERT(UInt8, INFINI_DTYPE_U8)
    CASE_DTYPE_CONVERT(UInt16, INFINI_DTYPE_U16)
    CASE_DTYPE_CONVERT(UInt32, INFINI_DTYPE_U32)
    CASE_DTYPE_CONVERT(UInt64, INFINI_DTYPE_U64)
    CASE_DTYPE_CONVERT(Bool, INFINI_DTYPE_BOOL)
    else {
        IT_TODO_HALT_MSG("Unsupported data type");
    }
}

#undef CASE_DTYPE_CONVERT // 使用后取消宏定义

} // namespace infini