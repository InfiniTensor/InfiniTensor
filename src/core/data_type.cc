#include "core/data_type.h"
#include "data_type.h" // InfiniOps data_type.h (resolved via -I.../infiniops/src)

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
// TODO: Reserved for complex data type.
const DataType DataType::BFloat16(16);

infini::ops::DataType toInfiniOpsDataType(const DataType &dt) {
    switch (dt.getIndex()) {
    case 1:  return infini::ops::DataType::kFloat32;
    case 2:  return infini::ops::DataType::kUInt8;
    case 3:  return infini::ops::DataType::kInt8;
    case 4:  return infini::ops::DataType::kUInt16;
    case 5:  return infini::ops::DataType::kInt16;
    case 6:  return infini::ops::DataType::kInt32;
    case 7:  return infini::ops::DataType::kInt64;
    case 10: return infini::ops::DataType::kFloat16;
    case 11: return infini::ops::DataType::kFloat64;
    case 12: return infini::ops::DataType::kUInt32;
    case 13: return infini::ops::DataType::kUInt64;
    case 16: return infini::ops::DataType::kBFloat16;
    default: IT_TODO_HALT_MSG("Unsupported DataType for InfiniOps");
    }
}

} // namespace infini
