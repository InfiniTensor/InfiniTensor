#include "operators/layer_norm.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class LayerNormXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LayerNormObj>(_op);
        auto context = static_cast<const KUNLUNRuntimeObj *>(_context);

        void *const inputData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const scaleData = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const outputData = (op->getOutput()->getRawDataPtr<void *>());

        float eps = op->getEps();
        // int axis = op->getAxis();

        auto opInputShape = op->getInputs(0)->getDims();
        auto opOutputShape = op->getOutput()->getDims();
        IT_ASSERT(opInputShape.size() == 2 || opInputShape.size() == 3);
        IT_ASSERT(opInputShape.size() == 3 && opInputShape[0] == 1);
        if (opInputShape.size() == 3) {
            opInputShape[0] = opInputShape[1];
            opInputShape[1] = opInputShape[2];
        }

        int ret;
        if (op->numInputs() == 3) {
            // with bias
            void *const biasData = op->getInputs(2)->getRawDataPtr<void *>();
            ret = xdnn::layer_norm<float, float>(
                context->KUNLUNHandle(), (float const *)inputData,
                (float *)outputData, opInputShape[0], opInputShape[1], eps,
                (float *)scaleData, (float *)biasData, nullptr, nullptr);
        } else {
            // without bias
            ret = xdnn::layer_norm<float, float>(
                context->KUNLUNHandle(), (float const *)inputData,
                (float *)outputData, opInputShape[0], opInputShape[1], eps,
                (float *)scaleData, nullptr, nullptr, nullptr);
        }
        assert(ret == 0);
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::LayerNormalization, LayerNormXdnn,
                "LayerNorm_xdnn_KUNLUN");

}; // namespace infini
