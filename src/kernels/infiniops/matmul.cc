#include "operators/matmul.h"
#include "infiniops_common.h"
#include "infiniops_dispatch.h"

#ifdef USE_INFINIOPS_ATEN_KERNELS
#include <base/add.h>
#include <base/matmul.h>
#endif

namespace infini {
namespace {

#ifdef USE_INFINIOPS_ATEN_KERNELS
size_t leadingProduct(const Shape &dims) {
    if (dims.size() <= 2)
        return 1;
    size_t product = 1;
    for (size_t i = 0; i + 2 < dims.size(); ++i) {
        IT_ASSERT(dims[i] >= 0);
        product *= static_cast<size_t>(dims[i]);
    }
    return product;
}

::infini::ops::Tensor makeMatmulInputView(const Tensor &tensor, size_t batch,
                                          bool transposed,
                                          const RuntimeObj *context) {
    const auto dims = tensor->getDims();
    IT_ASSERT(dims.size() >= 2);
    const auto physicalRows = static_cast<size_t>(dims[dims.size() - 2]);
    const auto physicalCols = static_cast<size_t>(dims[dims.size() - 1]);
    const auto leading = leadingProduct(dims);
    IT_ASSERT(leading == batch || leading == 1 || batch == 1);

    const auto rows = transposed ? physicalCols : physicalRows;
    const auto cols = transposed ? physicalRows : physicalCols;
    const auto rowStride = transposed ? size_t{1} : physicalCols;
    const auto colStride = transposed ? physicalCols : size_t{1};

    ::infini::rt::TensorView::Shape shape;
    ::infini::rt::TensorView::Strides strides;
    if (batch > 1) {
        shape = {batch, rows, cols};
        strides = {leading == 1 ? 0
                                : static_cast<::infini::rt::TensorView::Stride>(
                                      physicalRows * physicalCols),
                   static_cast<::infini::rt::TensorView::Stride>(rowStride),
                   static_cast<::infini::rt::TensorView::Stride>(colStride)};
    } else {
        shape = {rows, cols};
        strides = {static_cast<::infini::rt::TensorView::Stride>(rowStride),
                   static_cast<::infini::rt::TensorView::Stride>(colStride)};
    }

    return infiniops::makeInfiniOpsTensor(
        tensor->getRawDataPtr<void *>(), shape,
        infiniops::toInfiniOpsDataType(tensor->getDType()), context, strides);
}

::infini::ops::Tensor makeMatmulOutputView(const Tensor &tensor, size_t batch,
                                           const RuntimeObj *context) {
    const auto dims = tensor->getDims();
    IT_ASSERT(dims.size() >= 2);
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
    return batch > 1 ? Shape{batch, m, n} : Shape{m, n};
}

class MatmulInfiniOps : public infiniops::KernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(op->getAct() == ActType::None,
                  "Fused MatMul activation is not supported");

        const auto [batch, m, n, k] = op->getBMNK();
        (void)m;
        (void)n;
        (void)k;
        const auto batchSize = static_cast<size_t>(batch);
        auto inputA = makeMatmulInputView(op->getInputs(0), batchSize,
                                          op->getTransA(), context);
        auto inputB = makeMatmulInputView(op->getInputs(1), batchSize,
                                          op->getTransB(), context);
        auto output = makeMatmulOutputView(op->getOutput(), batchSize, context);
        auto handle = infiniops::makeInfiniOpsHandle(context);
        auto config =
            infiniops::makeInfiniOpsAtenConfig<::infini::ops::Matmul>(context);

        infiniops::dispatch::callMatmul(handle, config, inputA, inputB, output);
        if (op->getBias()) {
            auto bias = infiniops::toInfiniOpsBroadcastTensor(
                op->getBias(), flattenedOutputDims(*op), context);
            auto addConfig =
                infiniops::makeInfiniOpsConfig<::infini::ops::Add>(context);
            infiniops::dispatch::callAdd(handle, addConfig, output, bias, 1.0,
                                         output);
        }
    }
};
#endif

} // namespace

#ifdef USE_INFINIOPS_ATEN_KERNELS
REGISTER_KERNEL(ExecutionProvider::Infini, OpType::MatMul, MatmulInfiniOps,
                "Matmul_InfiniOps");
#endif

} // namespace infini
