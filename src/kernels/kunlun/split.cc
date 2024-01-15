#include "operators/split.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class SplitXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SplitObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        int axis = op->getDim();
        int num = op->numOutputs();
        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        auto inputDim = op->getInputs(0)->getDims();

        std::vector<float *> outputsData;
        for (int i = 0; i < num; ++i) {
            outputsData.push_back(
                (float *)(op->getOutput(i)->getRawDataPtr<void *>()));
        }

        std::vector<int> splitList;
        for (int i = 0; i < num; ++i) {
            auto dim = op->getOutput(i)->getDims();
            splitList.push_back(dim[axis]);
        }

        auto ret = baidu::xpu::api::split<float>(
            context->KUNLUNHandle(), (float *)inputData, outputsData, inputDim,
            splitList, axis);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Split, SplitXdnn, "Split_xdnn_KUNLUN");
}; // namespace infini
