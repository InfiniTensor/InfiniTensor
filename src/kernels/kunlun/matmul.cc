#include "operators/matmul.h"
#include "kunlun/kunlun_act_type.h"
#include "kunlun/kunlun_common.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
#include "utils/small_array.h"
#include "utils/operator_utils.h"

namespace infini {
class MatmulXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        // This kernel do C = act(alpha * x * w + beta * bias)
        auto op = as<MatmulObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
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


        // std::cout << vecToString<int>(aShape) << std::endl;
        // std::cout << vecToString<int>(bShape) << std::endl;
        // std::cout << vecToString<int>(cShape) << std::endl;
        // std::cout << b << ", "<< m << ", "<< n << ", "<< k << std::endl;

        float alpha = 1.f, beta = 0.f;
        Tensor biasTensor = op->getBias();
        KUNLUNPtr wkspace = nullptr;

        if (b > 1) {
            // Batch mul
            Tensor out = op->getOutput();
            size_t outSize = out->size();
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
                auto biasShape = biasTensor->getDims();
                auto gap = cShape.size() - biasShape.size();
                IT_ASSERT(gap >= 0);
                biasShape.insert(biasShape.begin(), gap, 1);
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

REGISTER_KERNEL(Device::KUNLUN, OpType::MatMul, MatmulXdnn,
                "Matmul_xdnn_KUNLUN");
}; // namespace infini
