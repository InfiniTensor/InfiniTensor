#include "data_type.h"

constexpr size_t DataType::size() const {
    switch (id) {
    case DataTypeId::FLOAT:
        return sizeof(float);
    case DataTypeId::UINT8:
        return sizeof(uint8_t);
    case DataTypeId::INT8:
        return sizeof(int8_t);
    case DataTypeId::UINT16:
        return sizeof(uint16_t);
    case DataTypeId::INT16:
        return sizeof(int16_t);
    case DataTypeId::INT32:
        return sizeof(int32_t);
    case DataTypeId::INT64:
        return sizeof(int64_t);
    case DataTypeId::BOOL:
        return sizeof(bool);
    case DataTypeId::FLOAT16:
        return 2;
    case DataTypeId::DOUBLE:
        return sizeof(double);
    case DataTypeId::UINT32:
        return sizeof(uint32_t);
    case DataTypeId::UINT64:
        return sizeof(uint64_t);
    default:
        throw "unsupported data type.";
    }
}
