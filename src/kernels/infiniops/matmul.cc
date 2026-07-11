#include "infiniops_common.h"
#include "operators/matmul.h"

#include <base/add.h>
#include <base/gemm.h>
#include <optional>

namespace infini {

namespace {

size_t leadingProduct(const Shape &dims) {
    if (dims.size() <= 2) {
        return 1;
    }

    size_t product = 1;
    for (size_t i = 0; i + 2 < dims.size(); ++i) {
        IT_ASSERT(dims[i] >= 0);
        product *= static_cast<size_t>(dims[i]);
    }
    return product;
}

::infini::ops::Tensor makeMatmulInputView(const Tensor &tensor, size_t batch,
                                          const RuntimeObj *context) {
    const auto dims = tensor->getDims();
    IT_ASSERT(dims.size() >= 2);
    IT_ASSERT(dims[dims.size() - 2] >= 0);
    IT_ASSERT(dims[dims.size() - 1] >= 0);

    const auto rows = static_cast<size_t>(dims[dims.size() - 2]);
    const auto cols = static_cast<size_t>(dims[dims.size() - 1]);
    const auto leading = leadingProduct(dims);
    IT_ASSERT(leading == batch || leading == 1 || batch == 1);

    ::infini::rt::TensorView::Shape shape;
    ::infini::rt::TensorView::Strides strides;
    if (batch > 1) {
        shape = {batch, rows, cols};
        strides = {leading == 1
                       ? 0
                       : static_cast<::infini::rt::TensorView::Stride>(rows *
                                                                       cols),
                   static_cast<::infini::rt::TensorView::Stride>(cols), 1};
    } else {
        shape = {rows, cols};
        strides = {static_cast<::infini::rt::TensorView::Stride>(cols), 1};
    }

    return infiniops::makeInfiniOpsTensor(
        tensor->getRawDataPtr<void *>(), shape,
        infiniops::toInfiniOpsDataType(tensor->getDType()), context, strides);
}

::infini::ops::Tensor makeMatmulOutputView(const Tensor &tensor, size_t batch,
                                           const RuntimeObj *context) {
    const auto dims = tensor->getDims();
    IT_ASSERT(dims.size() >= 2);
    IT_ASSERT(dims[dims.size() - 2] >= 0);
    IT_ASSERT(dims[dims.size() - 1] >= 0);

    const auto rows = static_cast<size_t>(dims[dims.size() - 2]);
    const auto cols = static_cast<size_t>(dims[dims.size() - 1]);
    IT_ASSERT(leadingProduct(dims) == batch);

    ::infini::rt::TensorView::Shape shape;
    ::infini::rt::TensorView::Strides strides;
    if (batch > 1) {
        shape = {batch, rows, cols};
        strides = {static_cast<::infini::rt::TensorView::Stride>(rows * cols),
                   static_cast<::infini::rt::TensorView::Stride>(cols), 1};
    } else {
        shape = {rows, cols};
        strides = {static_cast<::infini::rt::TensorView::Stride>(cols), 1};
    }

    return infiniops::makeInfiniOpsTensor(
        tensor->getRawDataPtr<void *>(), shape,
        infiniops::toInfiniOpsDataType(tensor->getDType()), context, strides);
}

Shape flattenedOutputDims(const MatmulObj &op) {
    const auto [batch, m, n, k] = op.getBMNK();
    (void)k;
    if (batch > 1) {
        return {batch, m, n};
    }
    return {m, n};
}

class MatmulInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(op->getAct() == ActType::None);

        const auto [batch, m, n, k] = op->getBMNK();
        (void)m;
        (void)n;
        (void)k;
        const auto batchSize = static_cast<size_t>(batch);

        auto inputA = makeMatmulInputView(op->getInputs(0), batchSize, context);
        auto inputB = makeMatmulInputView(op->getInputs(1), batchSize, context);
        auto output = makeMatmulOutputView(op->getOutput(), batchSize, context);

        auto handle = infiniops::makeInfiniOpsHandle(context);
        ::infini::ops::Gemm::Call(
            handle, {}, inputA, inputB, std::optional<float>{1.0f},
            std::optional<float>{0.0f},
            std::optional<int>{op->getTransA() ? 1 : 0},
            std::optional<int>{op->getTransB() ? 1 : 0}, output);

        if (op->getBias()) {
            const auto outDims = flattenedOutputDims(*op);
            auto bias = infiniops::toInfiniOpsBroadcastTensor(
                op->getBias(), outDims, context);
            ::infini::ops::Add::Call(handle, {}, output, bias, output);
        }
    }
};

} // namespace

#ifdef USE_CUDA
REGISTER_KERNEL(Device::CUDA, OpType::MatMul, MatmulInfiniOps,
                "Matmul_InfiniOps_CUDA");
#endif

} // namespace infini
