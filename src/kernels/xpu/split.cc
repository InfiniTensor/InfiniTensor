#include "operators/split.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class SplitXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SplitObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);
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
            if (dim.size() != 4) {
                IT_TODO_HALT();
            }
            splitList.push_back(dim[axis]);
        }

        auto ret = baidu::xpu::api::split<float>(
            context->XPUHandle(), (float *)inputData, outputsData, inputDim,
            splitList, axis);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::XPU, OpType::Split, DataType::Float32, SplitXdnn,
                "Split_xdnn_XPU_Float32");
}; // namespace infini
