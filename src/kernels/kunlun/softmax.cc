#include "operators/softmax.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class SoftmaxXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);
        auto dim = op->getInputs(0)->getDims();
        auto axis = op->getAxis();

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        size_t numSize = std::accumulate(dim.begin(), dim.end(), 1,
                                         std::multiplies<ShapeElem>()) /
                         dim.at(axis);
        KUNLUNPtr temp = context->getWorkspace(numSize * sizeof(float));

        // reduce_max
        checkKUNLUNError(
            baidu::xpu::api::reduce_max(context->KUNLUNHandle(), (float *)aData,
                                        (float *)temp, dim, Shape{axis}));
        auto temp2 = (float *)temp + numSize;
        Shape temp2Shape(dim);
        temp2Shape[axis] = 1;
        checkKUNLUNError(baidu::xpu::api::broadcast_sub(
            context->KUNLUNHandle(), (float *)aData, (float *)temp,
            (float *)temp2, dim, temp2Shape));

        auto ret = baidu::xpu::api::softmax<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData, dim, axis);
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Softmax, DataType::Float32, SoftmaxXdnn,
                "Softmax_xdnn_KUNLUN_Float32");
}; // namespace infini
