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
                    if (dtype == DataType::Float32) {
                        checkKUNLUNError(xdnn::broadcast<float>(
                            context->KUNLUNHandle(), (float *)aData,
                            (float *)wkspace, aDimsMatmul, aDimsTarget));
                    } else if (dtype == DataType::Float16) {
                        checkKUNLUNError(xdnn::broadcast<float16>(
                            context->KUNLUNHandle(), (float16 *)aData,
                            (float16 *)wkspace, aDimsMatmul, aDimsTarget));
                    } else if (dtype == DataType::Int32) {
                        checkKUNLUNError(xdnn::broadcast<int>(
                            context->KUNLUNHandle(), (int *)aData,
                            (int *)wkspace, aDimsMatmul, aDimsTarget));
                    } else if (dtype == DataType::Int8) {
                        checkKUNLUNError(xdnn::broadcast<int8_t>(
                            context->KUNLUNHandle(), (int8_t *)aData,
                            (int8_t *)wkspace, aDimsMatmul, aDimsTarget));
                    } else if (dtype == DataType::Int64) {
                        checkKUNLUNError(xdnn::broadcast<int64_t>(
                            context->KUNLUNHandle(), (int64_t *)aData,
                            (int64_t *)wkspace, aDimsMatmul, aDimsTarget));
                    } else if (dtype == DataType::Int16) {
                        checkKUNLUNError(xdnn::broadcast<int16_t>(
                            context->KUNLUNHandle(), (int16_t *)aData,
                            (int16_t *)wkspace, aDimsMatmul, aDimsTarget));
                    } else {
                        IT_ASSERT(false, "unsupported data type " +
                                             op->getDType().toString());
                    }

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
                    if (dtype == DataType::Float32) {
                        checkKUNLUNError(xdnn::broadcast<float>(
                            context->KUNLUNHandle(), (float *)bData,
                            (float *)wkspace, bDimsMatmul, bDimsTarget));
                    } else if (dtype == DataType::Float16) {
                        checkKUNLUNError(xdnn::broadcast<float16>(
                            context->KUNLUNHandle(), (float16 *)bData,
                            (float16 *)wkspace, bDimsMatmul, bDimsTarget));
                    } else if (dtype == DataType::Int32) {
                        checkKUNLUNError(xdnn::broadcast<int>(
                            context->KUNLUNHandle(), (int *)bData,
                            (int *)wkspace, bDimsMatmul, bDimsTarget));
                    } else if (dtype == DataType::Int8) {
                        checkKUNLUNError(xdnn::broadcast<int8_t>(
                            context->KUNLUNHandle(), (int8_t *)bData,
                            (int8_t *)wkspace, bDimsMatmul, bDimsTarget));
                    } else if (dtype == DataType::Int64) {
                        checkKUNLUNError(xdnn::broadcast<int64_t>(
                            context->KUNLUNHandle(), (int64_t *)bData,
                            (int64_t *)wkspace, bDimsMatmul, bDimsTarget));
                    } else if (dtype == DataType::Int16) {
                        checkKUNLUNError(xdnn::broadcast<int16_t>(
                            context->KUNLUNHandle(), (int16_t *)bData,
                            (int16_t *)wkspace, bDimsMatmul, bDimsTarget));
                    } else {
                        IT_ASSERT(false, "unsupported data type " +
                                             op->getDType().toString());
                    }

                    AData = aData;
                    BData = wkspace;
                    CData =
                        biasTensor
                            ? context->getWorkspace(numOutput * dtype.getSize())
                            : outData;
                } // endif batchA == 1
            } else { // batchA == batchB, no need to broadcast
                AData = aData;
                BData = bData;
                CData = biasTensor
                            ? context->getWorkspace(numOutput * dtype.getSize())
                            : outData;
            }
            if (op->getDType() == DataType::Float32) {
                checkKUNLUNError((xdnn::fc_batched<float, float, float, float>(
                    context->KUNLUNHandle(), b, transA, transB, m, n, k, alpha,
                    (float *)AData, m * k, (float *)BData, n * k, beta,
                    (float *)CData, m * n, nullptr, nullptr)));
            } else if (op->getDType() == DataType::Float16) {
                checkKUNLUNError(
                    (xdnn::fc_batched<float16, float16, float16, int16_t>(
                        context->KUNLUNHandle(), b, transA, transB, m, n, k,
                        alpha, (float16 *)AData, m * k, (float16 *)BData, n * k,
                        beta, (float16 *)CData, m * n, nullptr, nullptr)));
            } else {
                IT_ASSERT(false, "Unsupported data type: " +
                                     op->getDType().toString());
            }

            // Broadcast_add xw and bias if bias exists
            if (biasTensor) {
                auto biasShape = biasTensor->getDims();
                broadcastShape(cDims, biasShape);
                auto ret = 0;
                if (op->getDType() == DataType::Float32) {
                    ret = baidu::xpu::api::broadcast_add<float>(
                        context->KUNLUNHandle(), (float *)CData,
                        biasTensor->getRawDataPtr<float *>(), (float *)outData,
                        cDims, biasShape);
                } else if (op->getDType() == DataType::Float16) {
                    ret = baidu::xpu::api::broadcast_add<float16>(
                        context->KUNLUNHandle(), (float16 *)CData,
                        biasTensor->getRawDataPtr<float16 *>(),
                        (float16 *)outData, cDims, biasShape);
                } else if (op->getDType() == DataType::Int32) {
                    ret = baidu::xpu::api::broadcast_add<int>(
                        context->KUNLUNHandle(), (int *)CData,
                        biasTensor->getRawDataPtr<int *>(), (int *)outData,
                        cDims, biasShape);
                } else if (op->getDType() == DataType::Int64) {
                    ret = baidu::xpu::api::broadcast_add<int64_t>(
                        context->KUNLUNHandle(), (int64_t *)CData,
                        biasTensor->getRawDataPtr<int64_t *>(),
                        (int64_t *)outData, cDims, biasShape);
                } else {
                    IT_ASSERT(false, "Unsupported data type: " +
                                         op->getDType().toString());
                }
                checkKUNLUNError(ret);
            }
        } else {
            // Matmul with no batch, call fc_fusion
            const int lda = transA ? m : k, ldb = transB ? k : n, ldc = n;
            auto kunlunAct = parseActType(std::move(op->getAct()));
            if (op->getDType() == DataType::Float32) {
                checkKUNLUNError(
                    (baidu::xpu::api::fc_fusion<float, float, float, float>(
                        context->KUNLUNHandle(), (float *)aData, (float *)bData,
                        (float *)outData, m, n, k, transA, transB, nullptr,
                        nullptr, nullptr, lda, ldb, ldc, alpha, 0.f,
                        biasTensor ? biasTensor->getRawDataPtr<float *>()
                                   : nullptr,
                        kunlunAct, nullptr)));
            } else if (op->getDType() == DataType::Float16) {
                checkKUNLUNError((baidu::xpu::api::fc_fusion<float16, float16,
                                                             float16, int16_t>(
                    context->KUNLUNHandle(), (float16 *)aData, (float16 *)bData,
                    (float16 *)outData, m, n, k, transA, transB, nullptr,
                    nullptr, nullptr, lda, ldb, ldc, alpha, 0.f,
                    biasTensor ? biasTensor->getRawDataPtr<float *>() : nullptr,
                    kunlunAct, nullptr)));
            } else {
                IT_ASSERT(false, "Unsupported data type: " +
                                     op->getDType().toString());
            }
        }
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::MatMul, MatmulXdnn,
                "Matmul_xdnn_KUNLUN");
}; // namespace infini
