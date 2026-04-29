#pragma once

#include "core/data_type.h"
#include "core/tensor.h"
#include "core/tensor_base.h"
#include "data_type.h"
#include "device.h"
#include "handle.h"
#include "tensor.h"

namespace infini {

// Convert InfiniTensor DataType (class with index) to InfiniOps DataType (enum).
// InfiniTensor index mapping (from data_type.h names[]):
//   0=Undefine, 1=Float32, 2=UInt8, 3=Int8, 4=UInt16, 5=Int16,
//   6=Int32, 7=Int64, 8=String, 9=Bool, 10=Float16, 11=Double,
//   12=UInt32, 13=UInt64, 14-15=PlaceHolder, 16=BFloat16
inline infini::ops::DataType toInfiniOpsDataType(const DataType &dt) {
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

// Convert an InfiniTensor TensorObj to an InfiniOps Tensor (non-owning view).
// The returned Tensor holds a raw pointer to the TensorObj's data buffer;
// InfiniOps does NOT take ownership of the memory.
inline infini::ops::Tensor toInfiniOpsTensor(const TensorObj *tensor) {
    void *data = tensor->getRawDataPtr<void *>();

    // Shape: vector<int> → vector<size_t>
    auto dims = tensor->getDims();
    infini::ops::Tensor::Shape shape(dims.begin(), dims.end());

    auto dtype = toInfiniOpsDataType(tensor->getDType());

    auto device = tensor->getRuntime()->getDevice();

    // Strides: vector<int> → vector<ptrdiff_t>
    auto stride = tensor->getStride();
    infini::ops::Tensor::Strides strides(stride.begin(), stride.end());

    return infini::ops::Tensor(data, shape, dtype, device, strides);
}

} // namespace infini
