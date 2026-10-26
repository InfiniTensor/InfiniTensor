#include "infiniops_common.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/layer_norm.h"
#include "operators/unary.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/cast.h>
#include <base/div.h>
#include <base/erf.h>
#include <base/native_layer_norm.h>
#include <base/onnx_gather.h>
#include <base/sub.h>

namespace infini {
namespace {
template <typename Key, typename... Args>
void callEncoder(const RuntimeObj *context, const Args &...args) {
    auto config = infiniops::makeInfiniOpsAtenConfig<Key>(context);
    ::infini::ops::Operator<Key>::Call(infiniops::makeInfiniOpsHandle(context),
                                       config, args...);
}

class GatherInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<GatherObj>(_op);
        auto indexType = op->getInputs(1)->getDType();
        IT_ASSERT(indexType == DataType::Int64 || indexType == DataType::Int32,
                  "Gather indices must be Int32 or Int64");
        callEncoder<::infini::ops::OnnxGather>(
            context, infiniops::toInfiniOpsTensor(op->getInputs(0), context),
            infiniops::toInfiniOpsTensor(op->getInputs(1), context),
            int64_t(op->getAxis()),
            infiniops::toInfiniOpsTensor(op->getOutput(), context));
    }
};

class LayerNormInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<LayerNormObj>(_op);
        IT_ASSERT(
            op->getInputs(0)->getDType() == DataType::Float32 &&
                op->getStashType() == 1,
            "LayerNormalization bridge currently requires FP32/stash_type=1");
        auto dims = op->getInputs(0)->getDims();
        const auto axis = op->getAxis();
        Shape normalized(dims.begin() + axis, dims.end());
        std::vector<int64_t> normalizedShape(normalized.begin(),
                                             normalized.end());
        ::infini::rt::TensorView::Shape statsShape;
        size_t rows = 1;
        for (size_t i = 0; i < dims.size(); ++i) {
            auto size = i < size_t(axis) ? dims[i] : 1;
            statsShape.push_back(size);
            rows *= size;
        }
        auto meanBlob =
            infiniops::acquireWorkspace(context, rows * sizeof(float));
        auto rstdBlob =
            infiniops::acquireWorkspace(context, rows * sizeof(float));
        auto mean = infiniops::makeInfiniOpsTensor(
            meanBlob.get(), statsShape, ::infini::rt::DataType::kFloat32,
            context);
        auto rstd = infiniops::makeInfiniOpsTensor(
            rstdBlob.get(), statsShape, ::infini::rt::DataType::kFloat32,
            context);
        std::optional<::infini::ops::Tensor> weight =
            infiniops::toInfiniOpsBroadcastTensor(op->getInputs(1), normalized,
                                                  context);
        std::optional<::infini::ops::Tensor> bias;
        if (op->getBias())
            bias.emplace(infiniops::toInfiniOpsBroadcastTensor(
                op->getBias(), normalized, context));
        callEncoder<::infini::ops::NativeLayerNorm>(
            context, infiniops::toInfiniOpsTensor(op->getInputs(0), context),
            normalizedShape, weight, bias, double(op->getEps()),
            infiniops::toInfiniOpsTensor(op->getOutput(), context), mean, rstd);
    }
};

class EncoderUnaryInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto input = infiniops::toInfiniOpsTensor(op->getInputs(0), context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        switch (op->getOpType().underlying()) {
        case OpType::Cast:
            callEncoder<::infini::ops::Cast>(context, input, output);
            return;
        case OpType::Erf:
            callEncoder<::infini::ops::Erf>(context, input, output);
            return;
        default:
            IT_TODO_HALT_MSG("unsupported encoder unary operator");
        }
    }
};

class EncoderBinaryInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &op, const RuntimeObj *context) const override {
        auto dims = op->getOutput()->getDims();
        auto lhs = infiniops::toInfiniOpsBroadcastTensor(op->getInputs(0), dims,
                                                         context);
        auto rhs = infiniops::toInfiniOpsBroadcastTensor(op->getInputs(1), dims,
                                                         context);
        auto output = infiniops::toInfiniOpsTensor(op->getOutput(), context);
        switch (op->getOpType().underlying()) {
        case OpType::Sub:
            callEncoder<::infini::ops::Sub>(context, lhs, rhs, 1.0, output);
            return;
        case OpType::Div:
            if (op->getOutput()->getDType() == DataType::Int64 ||
                op->getOutput()->getDType() == DataType::Int32) {
                callEncoder<::infini::ops::Div>(
                    context, lhs, rhs, std::optional<std::string>{"trunc"},
                    output);
            } else {
                callEncoder<::infini::ops::Div>(context, lhs, rhs, output);
            }
            return;
        default:
            IT_TODO_HALT_MSG("unsupported encoder binary operator");
        }
    }
};
} // namespace
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Gather, GatherInfiniOps,
                "Gather_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::LayerNormalization,
                LayerNormInfiniOps, "LayerNorm_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Cast, EncoderUnaryInfiniOps,
                "Cast_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Erf, EncoderUnaryInfiniOps,
                "Erf_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Sub, EncoderBinaryInfiniOps,
                "Sub_InfiniOps");
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::Div, EncoderBinaryInfiniOps,
                "Div_InfiniOps");
} // namespace infini
#endif
