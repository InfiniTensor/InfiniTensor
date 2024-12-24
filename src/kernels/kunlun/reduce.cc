#include "operators/reduce.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {

class ReduceMeanXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceMeanObj>(_op);
        IT_ASSERT(op->getDType() == DataType::Float32);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto axes_set = op->getAxes();
        std::vector<int> axes;
        axes.assign(axes_set.begin(), axes_set.end());
        auto shape = op->getInputs(0)->getDims();

        auto ret = baidu::xpu::api::reduce_mean<float>(
            context->KUNLUNHandle(), (float *)aData, (float *)cData, shape,
            axes);
        assert(ret == 0);
        return;
    }
};

class ReduceSumXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceSumObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());

        auto axes_set = op->getAxes();
        std::vector<int> axes;
        axes.assign(axes_set.begin(), axes_set.end());
        auto shape = op->getInputs(0)->getDims();

        auto ret = 0;
        if (op->getDType() == DataType::Float32) {
            ret = baidu::xpu::api::reduce_sum<float>(
                context->KUNLUNHandle(), (float *)aData, (float *)cData, shape,
                axes);
        } else if (op->getDType() == DataType::Float16) {
            ret = baidu::xpu::api::reduce_sum<float16>(
                context->KUNLUNHandle(), (float16 *)aData, (float16 *)cData,
                shape, axes);
        } else if (op->getDType() == DataType::Int8) {
            ret = baidu::xpu::api::reduce_sum<int8_t>(
                context->KUNLUNHandle(), (int8_t *)aData, (int8_t *)cData,
                shape, axes);
        } else if (op->getDType() == DataType::Int32) {
            ret = baidu::xpu::api::reduce_sum<int>(context->KUNLUNHandle(),
                                                   (int *)aData, (int *)cData,
                                                   shape, axes);
        } else if (op->getDType() == DataType::Int64) {
            ret = baidu::xpu::api::reduce_sum<int64_t>(
                context->KUNLUNHandle(), (int64_t *)aData, (int64_t *)cData,
                shape, axes);
        } else {
            IT_ASSERT(false, "Unsupported data type");
        }

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::ReduceMean, ReduceMeanXdnn,
                "ReduceMean_xdnn_KUNLUN");
REGISTER_KERNEL(Device::KUNLUN, OpType::ReduceSum, ReduceSumXdnn,
                "ReduceSum_xdnn_KUNLUN");
}; // namespace infini
