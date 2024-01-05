#include "operators/matmul.h"
#include "kunlun/kunlun_act_type.h"
#include "kunlun/kunlun_common.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "utils/small_array.h"

namespace infini {
class MatmulXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        // This kernel do C = act(alpha * x * w + beta * bias)
        auto op = as<MatmulObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        Shape aShape = op->getInputs(0)->getDims();
        Shape bShape = op->getInputs(1)->getDims();
        Shape cShape = op->getOutput()->getDims();

        const auto [b, m, n, k] = op->getBMNK();

        bool transA = op->getTransA();
        bool transB = op->getTransB();
        float alpha = 1.f, beta = 0.f;
        Tensor biasTensor = op->getBias();
        KUNLUNPtr wkspace = nullptr;

        if (b > 1) {
            // Batch mul
            Tensor out = op->getOutput();
            size_t outSize = out->size();
            if (aShape.size() != bShape.size() || aShape[0] != bShape[0]) {
                // TODO: Batch Dimension need to be expanded
                IT_TODO_HALT_MSG("Batch dimension not equal");
            }
            if (biasTensor) { // If matmul with bias, need wkspace to do
                              // broadcast_add
                wkspace = context->getWorkspace(outSize *
                                                (out->getDType()).getSize());
            }
            // Calculate x * w
            checkKUNLUNError(
                (baidu::xpu::api::fc_batched<float, float, float, float>(
                    context->KUNLUNHandle(), b, transA, transB, m, n, k, 1.0,
                    (float *)aData, m * k, (float *)bData, n * k, beta,
                    // If bias exists, use wkspace to save internal results
                    (float *)(biasTensor ? wkspace : cData), m * n, nullptr,
                    nullptr)));
            // Broadcast_add xw and bias if bias exists
            if (biasTensor) {
                checkKUNLUNError(baidu::xpu::api::broadcast_add<float>(
                    context->KUNLUNHandle(), (float *)wkspace,
                    biasTensor->getRawDataPtr<float *>(), (float *)cData,
                    cShape, biasTensor->getDims()));
            }
        } else {
            // Matmul with no batch, call fc_fusion
            const int lda = transA ? m : k, ldb = transB ? k : n, ldc = n;
            auto kunlunAct = parseActType(std::move(op->getAct()));
            checkKUNLUNError(
                (baidu::xpu::api::fc_fusion<float, float, float, float>(
                    context->KUNLUNHandle(), (float *)aData, (float *)bData,
                    (float *)cData, m, n, k, transA, transB, nullptr, nullptr,
                    nullptr, lda, ldb, ldc, alpha, 0.f,
                    biasTensor ? biasTensor->getRawDataPtr<float *>() : nullptr,
                    kunlunAct, nullptr)));
        }
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::MatMul, DataType::Float32, MatmulXdnn,
                "Matmul_xdnn_KUNLUN_Float32");
}; // namespace infini
