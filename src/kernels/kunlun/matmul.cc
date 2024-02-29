#include "operators/matmul.h"
#include "kunlun/kunlun_act_type.h"
#include "kunlun/kunlun_common.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"
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
        void *const outData = (op->getOutput()->getRawDataPtr<void *>());

        Shape aDims = op->getInputs(0)->getDims();
        Shape bDims = op->getInputs(1)->getDims();
        Shape cDims = op->getOutput()->getDims();

        const auto [b, m, n, k] = op->getBMNK();
        bool transA = op->getTransA();
        bool transB = op->getTransB();
        int rankA = op->getInputs(0)->getRank();
        int rankB = op->getInputs(1)->getRank();
        int rankAligned = std::max(rankA, rankB);
        IT_ASSERT(rankAligned <= SMALL_ARRAY_SIZE);

        float alpha = 1.f, beta = 0.f;
        Tensor biasTensor = op->getBias();
        DataType dtype = op->getDType();

        if (b > 1) {
            SmallArray alignedAShape;
            SmallArray alignedBShape;
            // Padding 1 in aShape and bShape in order to align rank
            broadcastShape(aDims, alignedAShape, rankAligned, rankA);
            broadcastShape(bDims, alignedBShape, rankAligned, rankB);
            // Calculate batch dim
            int batchA = alignedAShape.prod(0, rankAligned - 2);
            int batchB = alignedBShape.prod(0, rankAligned - 2);
            // View aShape bShape to 3 dim
            Shape aDimsMatmul = {batchA, aDims[rankA - 2], aDims[rankA - 1]};
            Shape bDimsMatmul = {batchB, bDims[rankB - 2], bDims[rankB - 1]};
            auto numOutput = op->getOutput()->size();
            KUNLUNPtr wkspace = nullptr;
            void *AData = nullptr;
            void *BData = nullptr;
            void *CData = nullptr;
            if (batchA != batchB) {
                // If bs not equal, then broadcast
                IT_ASSERT(batchA == 1 || batchB == 1);
                if (batchA == 1) {
                    // Broadcast aShapeMatmul in batch dimension
                    Shape aDimsTarget = {b, aDimsMatmul[1], aDimsMatmul[2]};
                    auto numInput =
                        shapeProd(aDimsTarget.begin(), aDimsTarget.end());
                    wkspace = context->getWorkspace(numInput * dtype.getSize());
                    checkKUNLUNError(xdnn::broadcast<float>(
                        context->KUNLUNHandle(), (float *)aData,
                        (float *)wkspace, aDimsMatmul, aDimsTarget));
                    AData = wkspace;
                    BData = bData;
                    CData =
                        biasTensor
                            ? context->getWorkspace(numOutput * dtype.getSize())
                            : outData;
                } else {
                    // Broadcast bShapeMatmul in batch dimension
                    Shape bDimsTarget = {b, bDimsMatmul[1], bDimsMatmul[2]};
                    auto numInput =
                        shapeProd(bDimsTarget.begin(), bDimsTarget.end());
                    wkspace = context->getWorkspace(numInput * dtype.getSize());
                    checkKUNLUNError(xdnn::broadcast<float>(
                        context->KUNLUNHandle(), (float *)bData,
                        (float *)wkspace, bDimsMatmul, bDimsTarget));
                    AData = aData;
                    BData = wkspace;
                    CData =
                        biasTensor
                            ? context->getWorkspace(numOutput * dtype.getSize())
                            : outData;
                }    // endif batchA == 1
            } else { // batchA == batchB, no need to broadcast
                AData = aData;
                BData = bData;
                CData = biasTensor
                            ? context->getWorkspace(numOutput * dtype.getSize())
                            : outData;
            }
            checkKUNLUNError((xdnn::fc_batched<float, float, float, float>(
                context->KUNLUNHandle(), b, transA, transB, m, n, k, alpha,
                (float *)AData, m * k, (float *)BData, n * k, beta,
                (float *)CData, m * n, nullptr, nullptr)));
            // Broadcast_add xw and bias if bias exists
            if (biasTensor) {
                auto biasShape = biasTensor->getDims();
                broadcastShape(cDims, biasShape);
                checkKUNLUNError(baidu::xpu::api::broadcast_add<float>(
                    context->KUNLUNHandle(), (float *)CData,
                    biasTensor->getRawDataPtr<float *>(), (float *)outData,
                    cDims, biasShape));
            }
        } else {
            // Matmul with no batch, call fc_fusion
            const int lda = transA ? m : k, ldb = transB ? k : n, ldc = n;
            auto kunlunAct = parseActType(std::move(op->getAct()));
            checkKUNLUNError(
                (baidu::xpu::api::fc_fusion<float, float, float, float>(
                    context->KUNLUNHandle(), (float *)aData, (float *)bData,
                    (float *)outData, m, n, k, transA, transB, nullptr, nullptr,
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
