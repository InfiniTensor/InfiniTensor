#include "operators/batch_norm.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class BatchNormXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const input = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const mean = (op->getInputs(1)->getRawDataPtr<void *>());
        void *const var = (op->getInputs(2)->getRawDataPtr<void *>());
        void *const scale = (op->getInputs(3)->getRawDataPtr<void *>());
        void *const bias = (op->getInputs(4)->getRawDataPtr<void *>());
        void *const output = (op->getOutput()->getRawDataPtr<void *>());

        auto dims = op->getInputs(0)->getDims();

        if (dims.size() != 4)
            IT_TODO_HALT();

        int w = dims[3];
        int h = dims[2];
        int c = dims[1];
        int n = dims[0];
        auto ret = baidu::xpu::api::batch_norm_infer<float>(
            context->KUNLUNHandle(), (float *)input, (float *)output, n, c, h, w,
            op->getEps(), (float *)scale, (float *)bias, (float *)mean,
            (float *)var, true);

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::BatchNormalization, DataType::Float32,
                BatchNormXdnn, "BatchNorm_xdnn_KUNLUN_Float32");

}; // namespace infini
