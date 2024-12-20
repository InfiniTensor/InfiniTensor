#include "operators/split.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class SplitXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SplitObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        int axis = op->getDim();
        int num = op->numOutputs();
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        auto inputDim = op->getInputs(0)->getDims();

        std::vector<int> splitList;
        for (int i = 0; i < num; ++i) {
            auto dim = op->getOutput(i)->getDims();
            splitList.push_back(dim[axis]);
        }
        auto ret = 0;
        if (op->getDType() == DataType::Float32) {
            std::vector<float *> outputsData;
            for (int i = 0; i < num; ++i) {
                outputsData.push_back(
                    (float *)(op->getOutput(i)->getRawDataPtr<void *>()));
            }
            ret = baidu::xpu::api::split<float>(context->KUNLUNHandle(),
                                                (float *)inputData, outputsData,
                                                inputDim, splitList, axis);
        } else if (op->getDType() == DataType::Float16) {
            std::vector<float16 *> outputsData;
            for (int i = 0; i < num; ++i) {
                outputsData.push_back(
                    (float16 *)(op->getOutput(i)->getRawDataPtr<void *>()));
            }
            ret = baidu::xpu::api::split<float16>(
                context->KUNLUNHandle(), (float16 *)inputData, outputsData,
                inputDim, splitList, axis);
        } else if (op->getDType() == DataType::Int8) {
            std::vector<int8_t *> outputsData;
            for (int i = 0; i < num; ++i) {
                outputsData.push_back(
                    (int8_t *)(op->getOutput(i)->getRawDataPtr<void *>()));
            }
            ret = baidu::xpu::api::split<int8_t>(
                context->KUNLUNHandle(), (int8_t *)inputData, outputsData,
                inputDim, splitList, axis);
        } else if (op->getDType() == DataType::Int32) {
            std::vector<int *> outputsData;
            for (int i = 0; i < num; ++i) {
                outputsData.push_back(
                    (int *)(op->getOutput(i)->getRawDataPtr<void *>()));
            }
            ret = baidu::xpu::api::split<int>(context->KUNLUNHandle(),
                                              (int *)inputData, outputsData,
                                              inputDim, splitList, axis);
        } else if (op->getDType() == DataType::Int64) {
            std::vector<int64_t *> outputsData;
            for (int i = 0; i < num; ++i) {
                outputsData.push_back(
                    (int64_t *)(op->getOutput(i)->getRawDataPtr<void *>()));
            }
            ret = baidu::xpu::api::split<int64_t>(
                context->KUNLUNHandle(), (int64_t *)inputData, outputsData,
                inputDim, splitList, axis);
        } else if (op->getDType() == DataType::Int16) {
            std::vector<int16_t *> outputsData;
            for (int i = 0; i < num; ++i) {
                outputsData.push_back(
                    (int16_t *)(op->getOutput(i)->getRawDataPtr<void *>()));
            }
            ret = baidu::xpu::api::split<int16_t>(
                context->KUNLUNHandle(), (int16_t *)inputData, outputsData,
                inputDim, splitList, axis);
        } else {
            IT_ASSERT(false, "Unsupported data type");
        }

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Split, SplitXdnn, "Split_xdnn_KUNLUN");
}; // namespace infini
