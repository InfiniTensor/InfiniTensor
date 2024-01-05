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
        std::vector<const float *> inputsData;
        for (int i = 0; i < num; ++i) {
            inputsData.push_back(
                (float *)(op->getInputs(i)->getRawDataPtr<void *>()));
        }
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        std::vector<std::vector<int>> dims;
        for (int i = 0; i < num; ++i) {
            auto dim = op->getInputs(i)->getDims();
            if (dim.size() != 4) {
                IT_TODO_HALT();
            }
            dims.push_back(dim);
        }
        auto ret = xdnn::concat<float>(
            context->KUNLUNHandle(), inputsData, (float *)cData, dims, axis);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Concat, DataType::Float32, ConcatXdnn,
                "Concat_xdnn_KUNLUN_Float32");
}; // namespace infini
