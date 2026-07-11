#include "infiniops_common.h"
#include "operators/rope.h"

#include <algorithm>
#include <base/rotary_embedding_infinilm.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace infini {

namespace {

size_t readMaxPosition(const Tensor &pos, const RuntimeObj *context) {
    IT_ASSERT(pos != nullptr);
    size_t maxPos = 0;
    const auto count = pos->size();

    if (pos->getDType() == DataType::Int32) {
        std::vector<int32_t> host(count);
        context->copyBlobToCPU(host.data(), pos->getRawDataPtr<void *>(),
                               pos->getBytes());
        for (auto value : host) {
            IT_ASSERT(value >= 0);
            maxPos = std::max(maxPos, static_cast<size_t>(value));
        }
        return maxPos;
    }

    if (pos->getDType() == DataType::Int64) {
        std::vector<int64_t> host(count);
        context->copyBlobToCPU(host.data(), pos->getRawDataPtr<void *>(),
                               pos->getBytes());
        for (auto value : host) {
            IT_ASSERT(value >= 0);
            maxPos = std::max(maxPos, static_cast<size_t>(value));
        }
        return maxPos;
    }

    IT_TODO_HALT_MSG("RoPE position ids must be int32 or int64");
}

void buildRopeTables(size_t tableLen, size_t tableDim, size_t headDim,
                     std::vector<float> &sinTable,
                     std::vector<float> &cosTable) {
    sinTable.resize(tableLen * tableDim);
    cosTable.resize(tableLen * tableDim);

    for (size_t pos = 0; pos < tableLen; ++pos) {
        for (size_t i = 0; i < tableDim; ++i) {
            const auto invFreq =
                std::pow(10000.0f,
                         -static_cast<float>(2 * i) /
                             static_cast<float>(headDim));
            const auto angle = static_cast<float>(pos) * invFreq;
            sinTable[pos * tableDim + i] = std::sin(angle);
            cosTable[pos * tableDim + i] = std::cos(angle);
        }
    }
}

class RoPEInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<RoPEObj>(_op);
        auto pos = op->getInputs(0);
        auto input = op->getInputs(1);
        auto output = op->getOutput();

        const auto inputDims = input->getDims();
        const auto posDims = pos->getDims();
        const auto outputDims = output->getDims();
        IT_ASSERT(inputDims.size() == 3);
        IT_ASSERT(posDims.size() == 2);
        IT_ASSERT(outputDims == inputDims);
        IT_ASSERT(inputDims[0] == posDims[0]);
        IT_ASSERT(inputDims[1] == posDims[1]);

        const size_t batch = static_cast<size_t>(inputDims[0]);
        const size_t seqLen = static_cast<size_t>(inputDims[1]);
        const size_t hidden = static_cast<size_t>(inputDims[2]);
        const size_t headDim = 128;
        IT_ASSERT(hidden % headDim == 0);
        const size_t numHeads = hidden / headDim;
        const size_t tableDim = headDim / 2;

        auto maxPos = readMaxPosition(pos, context);
        auto tableLen = std::max(seqLen, maxPos + 1);
        std::vector<float> sinHost;
        std::vector<float> cosHost;
        buildRopeTables(tableLen, tableDim, headDim, sinHost, cosHost);

        const auto tableBytes = tableLen * tableDim * sizeof(float);
        auto sinBlob = infiniops::allocTemporaryBlob(context, tableBytes);
        auto cosBlob = infiniops::allocTemporaryBlob(context, tableBytes);
        context->copyBlobFromCPU(sinBlob->getPtr<void *>(), sinHost.data(),
                                 tableBytes);
        context->copyBlobFromCPU(cosBlob->getPtr<void *>(), cosHost.data(),
                                 tableBytes);

        ::infini::rt::TensorView::Shape ropeShape{batch, seqLen, numHeads,
                                                  headDim};
        ::infini::rt::TensorView::Strides ropeStrides{
            static_cast<::infini::rt::TensorView::Stride>(seqLen * hidden),
            static_cast<::infini::rt::TensorView::Stride>(hidden),
            static_cast<::infini::rt::TensorView::Stride>(headDim),
            1};
        ::infini::rt::TensorView::Shape tableShape{tableLen, tableDim};

        auto inputView = infiniops::makeInfiniOpsTensor(
            input->getRawDataPtr<void *>(), ropeShape,
            infiniops::toInfiniOpsDataType(input->getDType()), context,
            ropeStrides);
        auto posView = infiniops::toInfiniOpsTensor(pos, context);
        auto sinView = infiniops::makeInfiniOpsTensor(
            sinBlob->getPtr<void *>(), tableShape,
            ::infini::rt::DataType::kFloat32, context);
        auto cosView = infiniops::makeInfiniOpsTensor(
            cosBlob->getPtr<void *>(), tableShape,
            ::infini::rt::DataType::kFloat32, context);
        auto outputView = infiniops::makeInfiniOpsTensor(
            output->getRawDataPtr<void *>(), ropeShape,
            infiniops::toInfiniOpsDataType(output->getDType()), context,
            ropeStrides);

        auto handle = infiniops::makeInfiniOpsHandle(context);
        ::infini::ops::RotaryEmbeddingInfinilm::Call(
            handle, {}, inputView, posView, sinView, cosView, false,
            outputView);
    }
};

} // namespace

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::RoPE, RoPEInfiniOps,
                "RoPE_InfiniOps_CUDA");
#endif

} // namespace infini
