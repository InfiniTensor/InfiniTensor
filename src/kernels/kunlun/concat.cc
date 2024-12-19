#include "operators/concat.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class ConcatXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConcatObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        int axis = op->getDim();
        int num = op->numInputs();

        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        std::vector<std::vector<int>> dims;
        for (int i = 0; i < num; ++i) {
            auto dim = op->getInputs(i)->getDims();
            dims.push_back(dim);
        }
        auto ret = 0;
        if (op->getDType() == DataType::Float32) {
            std::vector<const float *> inputsData;
            for (int i = 0; i < num; ++i) {
                inputsData.push_back(
                    (float *)(op->getInputs(i)->getRawDataPtr<void *>()));
            }
            ret = xdnn::concat<float>(context->KUNLUNHandle(), inputsData,
                                      (float *)cData, dims, axis);
        } else if (op->getDType() == DataType::Float16) {
            std::vector<const float16 *> inputsData;
            for (int i = 0; i < num; ++i) {
                inputsData.push_back(
                    (float16 *)(op->getInputs(i)->getRawDataPtr<void *>()));
            }
            ret = xdnn::concat<float16>(context->KUNLUNHandle(), inputsData,
                                        (float16 *)cData, dims, axis);
        } else if (op->getDType() == DataType::Int8) {
            std::vector<const int8_t *> inputsData;
            for (int i = 0; i < num; ++i) {
                inputsData.push_back(
                    (int8_t *)(op->getInputs(i)->getRawDataPtr<void *>()));
            }
            ret = xdnn::concat<int8_t>(context->KUNLUNHandle(), inputsData,
                                       (int8_t *)cData, dims, axis);
        } else if (op->getDType() == DataType::Int32) {
            std::vector<const int *> inputsData;
            for (int i = 0; i < num; ++i) {
                inputsData.push_back(
                    (int *)(op->getInputs(i)->getRawDataPtr<void *>()));
            }
            ret = xdnn::concat<int>(context->KUNLUNHandle(), inputsData,
                                    (int *)cData, dims, axis);
        } else if (op->getDType() == DataType::Int64) {
            std::vector<const int64_t *> inputsData;
            for (int i = 0; i < num; ++i) {
                inputsData.push_back(
                    (int64_t *)(op->getInputs(i)->getRawDataPtr<void *>()));
            }
            ret = xdnn::concat<int64_t>(context->KUNLUNHandle(), inputsData,
                                        (int64_t *)cData, dims, axis);
        } else if (op->getDType() == DataType::Int16) {
            std::vector<const int16_t *> inputsData;
            for (int i = 0; i < num; ++i) {
                inputsData.push_back(
                    (int16_t *)(op->getInputs(i)->getRawDataPtr<void *>()));
            }
            ret = xdnn::concat<int16_t>(context->KUNLUNHandle(), inputsData,
                                        (int16_t *)cData, dims, axis);
        } else {
            IT_ASSERT(false, "Unsupported data type");
        }

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Concat, ConcatXdnn,
                "Concat_xdnn_KUNLUN");
}; // namespace infini

