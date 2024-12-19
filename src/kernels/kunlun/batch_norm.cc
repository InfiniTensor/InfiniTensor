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

        int n, c, h, w;
        if (dims.size() != 4) {
            h = 1;
            w = 1;
        }

        w = dims[3];
        h = dims[2];
        c = dims[1];
        n = dims[0];

        if (op->getDType() == DataType::Float32) {
            checkKUNLUNError(xdnn::batch_norm_infer<float>(
                context->KUNLUNHandle(), (float *)input, (float *)output, n, c,
                h, w, op->getEps(), (float *)scale, (float *)bias,
                (float *)mean, (float *)var, true));
        } else if (op->getDType() == DataType::Float16) {
            checkKUNLUNError(xdnn::batch_norm_infer<float16>(
                context->KUNLUNHandle(), (float16 *)input, (float16 *)output, n,
                c, h, w, op->getEps(), (float *)scale, (float *)bias,
                (float *)mean, (float *)var, true));
        } else {
            IT_ASSERT(false,
                      "unsupported data type " + op->getDType().toString());
        }

        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::BatchNormalization, BatchNormXdnn,
                "BatchNorm_xdnn_KUNLUN");

}; // namespace infini
