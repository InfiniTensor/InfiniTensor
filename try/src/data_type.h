#pragma once

#include <cstddef>
#include <cstdint>

enum class DataTypeId {
    UNDEFINED,
    FLOAT,
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    INT64,
    STRING,
    BOOL,
    FLOAT16,
    DOUBLE,
    UINT32,
    UINT64,
    // COMPLEX64,
    // COMPLEX128,
    // BFLOAT16,
};

struct DataType {
    DataTypeId id;

    size_t size() const;
};
