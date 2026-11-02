#pragma once

#include "core/blob.h"
#include "core/infini_runtime.h"
#include "core/kernel.h"

#include <infini/ops.h>

#include <algorithm>
#include <utility>

namespace infini {
namespace infiniops {

inline ::infini::rt::DataType
toInfiniOpsDataType(const ::infini::DataType &dtype) {
    using Type = ::infini::rt::DataType;
    static const std::pair<::infini::DataType, Type> types[]{
        {::infini::DataType::Bool, Type::kBool},
        {::infini::DataType::Float32, Type::kFloat32},
        {::infini::DataType::Float16, Type::kFloat16},
        {::infini::DataType::BFloat16, Type::kBFloat16},
        {::infini::DataType::Double, Type::kFloat64},
        {::infini::DataType::Int8, Type::kInt8},
        {::infini::DataType::Int16, Type::kInt16},
        {::infini::DataType::Int32, Type::kInt32},
        {::infini::DataType::Int64, Type::kInt64},
        {::infini::DataType::UInt8, Type::kUInt8},
        {::infini::DataType::UInt16, Type::kUInt16},
        {::infini::DataType::UInt32, Type::kUInt32},
        {::infini::DataType::UInt64, Type::kUInt64},
    };
    for (const auto &[source, target] : types)
        if (dtype == source)
            return target;
    IT_TODO_HALT_MSG("Unsupported InfiniTensor dtype for InfiniOps");
}

inline ::infini::rt::Device toInfiniOpsDevice(const RuntimeObj *context) {
    IT_ASSERT(context != nullptr);
    if (context->isCpu()) {
        return {::infini::rt::Device::Type::kCpu, context->getDeviceId()};
    }
    auto runtime = dynamic_cast<const InfiniRuntimeObj *>(context);
    IT_ASSERT(runtime != nullptr,
              "InfiniOps requires an InfiniRuntime or native CPU runtime");
    return runtime->getInfiniDevice();
}

inline ::infini::ops::Tensor toInfiniOpsTensor(const Tensor &tensor,
                                               const RuntimeObj *context) {
    IT_ASSERT(tensor != nullptr);
    const auto &dims = tensor->getDims();
    const auto strides = tensor->getStride();
    IT_ASSERT(
        std::all_of(dims.begin(), dims.end(), [](int d) { return d >= 0; }));
    return {tensor->getRawDataPtr<void *>(),
            ::infini::ops::Tensor::Shape(dims.begin(), dims.end()),
            toInfiniOpsDataType(tensor->getDType()), toInfiniOpsDevice(context),
            ::infini::ops::Tensor::Strides(strides.begin(), strides.end())};
}

inline ::infini::ops::Tensor
makeInfiniOpsTensor(void *data, const ::infini::rt::TensorView::Shape &shape,
                    const ::infini::rt::DataType &dtype,
                    const RuntimeObj *context,
                    const ::infini::rt::TensorView::Strides &strides = {}) {
    const auto device = toInfiniOpsDevice(context);
    return strides.empty()
               ? ::infini::ops::Tensor{data, shape, dtype, device}
               : ::infini::ops::Tensor{data, shape, dtype, device, strides};
}

inline ::infini::ops::Tensor
toInfiniOpsBroadcastTensor(const Tensor &tensor, const Shape &outputDims,
                           const RuntimeObj *context) {
    IT_ASSERT(tensor != nullptr);
    auto dims = tensor->getDims();
    auto strides = tensor->getStride();
    IT_ASSERT(dims.size() <= outputDims.size());

    ::infini::ops::Tensor::Shape shape(outputDims.begin(), outputDims.end());
    ::infini::ops::Tensor::Strides rtStrides(outputDims.size(), 0);
    const auto rankOffset = outputDims.size() - dims.size();
    for (size_t i = 0; i < outputDims.size(); ++i) {
        IT_ASSERT(outputDims[i] >= 0);
        if (i < rankOffset)
            continue;
        const auto j = i - rankOffset;
        IT_ASSERT(dims[j] == outputDims[i] || dims[j] == 1);
        if (dims[j] == outputDims[i])
            rtStrides[i] = strides[j];
    }

    return {tensor->getRawDataPtr<void *>(), shape,
            toInfiniOpsDataType(tensor->getDType()), toInfiniOpsDevice(context),
            rtStrides};
}

} // namespace infiniops
} // namespace infini
