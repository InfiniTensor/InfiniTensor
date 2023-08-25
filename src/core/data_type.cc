#include "core/data_type.h"

namespace infini {
// Move implementation here to avoid compile time error on some platform
// to be consistent with onnx
// https://github.com/onnx/onnx/blob/aeb21329122b96df1d3ef33b500a35ca140b1431/onnx/onnx.proto#L484
const DataType DataType::Undefine(0);
const DataType DataType::Float32(1);
const DataType DataType::UInt8(2);
const DataType DataType::Int8(3);
const DataType DataType::UInt16(4);
const DataType DataType::Int16(5);
const DataType DataType::Int32(6);
const DataType DataType::Int64(7);
const DataType DataType::String(8);
const DataType DataType::Bool(9);
const DataType DataType::Float16(10);
const DataType DataType::Double(11);
const DataType DataType::UInt32(12);
const DataType DataType::UInt64(13);
const DataType DataType::BFloat16(16);
} // namespace infini
