#include "operators/transpose.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class TransposeXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto dimin = op->getInputs(0)->getDims();
        auto permute = op->getPermute();

        auto ret = 0;
        if (op->getDType() == DataType::Float32) {
            ret =
                xdnn::transpose<float>(context->KUNLUNHandle(), (float *)aData,
                                       (float *)cData, dimin, permute);
        } else if (op->getDType() == DataType::Float16) {
            ret = xdnn::transpose<float16>(context->KUNLUNHandle(),
                                           (float16 *)aData, (float16 *)cData,
                                           dimin, permute);
        } else if (op->getDType() == DataType::Int8) {
            ret = xdnn::transpose<int8_t>(context->KUNLUNHandle(),
                                          (int8_t *)aData, (int8_t *)cData,
                                          dimin, permute);
        } else if (op->getDType() == DataType::Int32) {
            ret = xdnn::transpose<int>(context->KUNLUNHandle(), (int *)aData,
                                       (int *)cData, dimin, permute);
        } else if (op->getDType() == DataType::Int64) {
            ret = xdnn::transpose<int64_t>(context->KUNLUNHandle(),
                                           (int64_t *)aData, (int64_t *)cData,
                                           dimin, permute);
        } else if (op->getDType() == DataType::Int16) {
            ret = xdnn::transpose<int16_t>(context->KUNLUNHandle(),
                                           (int16_t *)aData, (int16_t *)cData,
                                           dimin, permute);
        } else {
            IT_ASSERT(false,
                      "unsupported data type " + op->getDType().toString());
        }
        assert(ret == 0);
        return;
    }
};

class DepthToSpaceXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<DepthToSpaceObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto reshape = op->getReshapeDim();
        auto mode = op->getMode();
        std::vector<int> permute;
        if (mode == 0) {
            permute = {0, 3, 4, 1, 5, 2};
        } else {
            permute = {0, 1, 4, 2, 5, 3};
        }
        auto ret =
            xdnn::transpose<float>(context->KUNLUNHandle(), (float *)aData,
                                   (float *)cData, reshape, permute);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Transpose, TransposeXdnn,
                "Transpose_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::DepthToSpace, DepthToSpaceXdnn,
                "DepthToSpace_xdnn_KUNLUN");
}; // namespace infini
