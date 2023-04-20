#pragma once

#include <cstddef>
#include <cstdint>

namespace optimization {

enum class DataTypeId : uint8_t {
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

template <class t> DataType ty();
template <> inline DataType ty<float>() { return {DataTypeId::FLOAT}; }
template <> inline DataType ty<uint8_t>() { return {DataTypeId::UINT8}; }
template <> inline DataType ty<int8_t>() { return {DataTypeId::INT8}; }
template <> inline DataType ty<uint16_t>() { return {DataTypeId::UINT16}; }
template <> inline DataType ty<int16_t>() { return {DataTypeId::INT16}; }
template <> inline DataType ty<int32_t>() { return {DataTypeId::INT32}; }
template <> inline DataType ty<int64_t>() { return {DataTypeId::INT64}; }
template <> inline DataType ty<bool>() { return {DataTypeId::BOOL}; }
template <> inline DataType ty<double>() { return {DataTypeId::DOUBLE}; }
template <> inline DataType ty<uint32_t>() { return {DataTypeId::UINT32}; }
template <> inline DataType ty<uint64_t>() { return {DataTypeId::UINT64}; }

} // namespace optimization
