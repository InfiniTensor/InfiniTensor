#pragma once

#include "core/blob.h"
#include "core/infini_runtime.h"
#include "core/kernel.h"

#include <infini/ops.h>

#include <memory>

namespace infini {
namespace infiniops {

inline ::infini::rt::DataType
toInfiniOpsDataType(const ::infini::DataType &dtype) {
    if (dtype == ::infini::DataType::Bool)
        return ::infini::rt::DataType::kBool;
    if (dtype == ::infini::DataType::Float32) {
        return ::infini::rt::DataType::kFloat32;
    }
    if (dtype == ::infini::DataType::Float16) {
        return ::infini::rt::DataType::kFloat16;
    }
    if (dtype == ::infini::DataType::BFloat16) {
        return ::infini::rt::DataType::kBFloat16;
    }
    if (dtype == ::infini::DataType::Double) {
        return ::infini::rt::DataType::kFloat64;
    }
    if (dtype == ::infini::DataType::Int8) {
        return ::infini::rt::DataType::kInt8;
    }
    if (dtype == ::infini::DataType::Int16) {
        return ::infini::rt::DataType::kInt16;
    }
    if (dtype == ::infini::DataType::Int32) {
        return ::infini::rt::DataType::kInt32;
    }
    if (dtype == ::infini::DataType::Int64) {
        return ::infini::rt::DataType::kInt64;
    }
    if (dtype == ::infini::DataType::UInt8) {
        return ::infini::rt::DataType::kUInt8;
    }
    if (dtype == ::infini::DataType::UInt16) {
        return ::infini::rt::DataType::kUInt16;
    }
    if (dtype == ::infini::DataType::UInt32) {
        return ::infini::rt::DataType::kUInt32;
    }
    if (dtype == ::infini::DataType::UInt64) {
        return ::infini::rt::DataType::kUInt64;
    }
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
    auto dims = tensor->getDims();
    auto strides = tensor->getStride();
    ::infini::rt::TensorView::Shape shape;
    ::infini::rt::TensorView::Strides rtStrides;
    shape.resize(dims.size());
    rtStrides.resize(strides.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        const auto dim = dims[i];
        IT_ASSERT(dim >= 0);
        shape[i] = static_cast<::infini::rt::TensorView::Size>(dim);
    }
    for (size_t i = 0; i < strides.size(); ++i) {
        rtStrides[i] =
            static_cast<::infini::rt::TensorView::Stride>(strides[i]);
    }
    return {tensor->getRawDataPtr<void *>(), shape,
            toInfiniOpsDataType(tensor->getDType()), toInfiniOpsDevice(context),
            rtStrides};
}

inline ::infini::ops::Tensor
makeInfiniOpsTensor(void *data, const ::infini::rt::TensorView::Shape &shape,
                    const ::infini::rt::DataType &dtype,
                    const RuntimeObj *context,
                    const ::infini::rt::TensorView::Strides &strides = {}) {
    auto rtStrides = strides;
    if (rtStrides.empty()) {
        rtStrides.resize(shape.size());
        ::infini::rt::TensorView::Stride stride = 1;
        for (auto i = shape.size(); i > 0; --i) {
            rtStrides[i - 1] = stride;
            stride *=
                static_cast<::infini::rt::TensorView::Stride>(shape[i - 1]);
        }
    }
    return {data, shape, dtype, toInfiniOpsDevice(context), rtStrides};
}

inline ::infini::ops::Tensor
toInfiniOpsBroadcastTensor(const Tensor &tensor, const Shape &outputDims,
                           const RuntimeObj *context) {
    IT_ASSERT(tensor != nullptr);
    auto dims = tensor->getDims();
    auto strides = tensor->getStride();
    IT_ASSERT(dims.size() <= outputDims.size());

    ::infini::rt::TensorView::Shape shape;
    ::infini::rt::TensorView::Strides rtStrides;
    shape.resize(outputDims.size());
    rtStrides.resize(outputDims.size());

    const auto rankOffset = outputDims.size() - dims.size();
    for (size_t i = 0; i < outputDims.size(); ++i) {
        const auto outputDim = outputDims[i];
        IT_ASSERT(outputDim >= 0);
        shape[i] = static_cast<::infini::rt::TensorView::Size>(outputDim);

        if (i < rankOffset) {
            rtStrides[i] = 0;
            continue;
        }

        const auto inputDim = dims[i - rankOffset];
        IT_ASSERT(inputDim == outputDim || inputDim == 1);
        if (inputDim == outputDim) {
            rtStrides[i] = static_cast<::infini::rt::TensorView::Stride>(
                strides[i - rankOffset]);
        } else {
            rtStrides[i] = 0;
        }
    }

    return {tensor->getRawDataPtr<void *>(), shape,
            toInfiniOpsDataType(tensor->getDType()), toInfiniOpsDevice(context),
            rtStrides};
}

} // namespace infiniops
} // namespace infini
