#include "operators/unary.h"
#include "xpu/xpu_kernel_without_config.h"
#include "xpu/xpu_runtime.h"

namespace infini {
class CastXdnn : public XPUKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<CastObj>(_op);
        auto context = dynamic_cast<const XPURuntimeObj *>(_context);

        void *const aData = (op->getInputs(0)->getRawDataPtr<void *>());
        void *const cData = (op->getOutput()->getRawDataPtr<void *>());
        auto len = op->getInputs(0)->size();
        CastType type = op->getType();

        int ret = 0;
        switch (type) {
        case CastType::Float2Float16:
            ret = baidu::xpu::api::cast<float, float16>(
                context->XPUHandle(), (float *)aData, (float16 *)cData, len);
            break;
        case CastType::Float2Int64:
            ret = baidu::xpu::api::cast<float, int64_t>(
                context->XPUHandle(), (float *)aData, (int64_t *)cData, len);
            break;
        case CastType::Float2Int32:
            ret = baidu::xpu::api::cast<float, int>(
                context->XPUHandle(), (float *)aData, (int *)cData, len);
            break;
        case CastType::Float2Int16:
            ret = baidu::xpu::api::cast<float, int16_t>(
                context->XPUHandle(), (float *)aData, (int16_t *)cData, len);
            break;
        case CastType::Float2Int8:
            ret = baidu::xpu::api::cast<float, int8_t>(
                context->XPUHandle(), (float *)aData, (int8_t *)cData, len);
            break;
        case CastType::Int322Float:
            ret = baidu::xpu::api::cast<int, float>(
                context->XPUHandle(), (int *)aData, (float *)cData, len);
            break;
        case CastType::Int322Int8:
            ret = baidu::xpu::api::cast<int, int8_t>(
                context->XPUHandle(), (int *)aData, (int8_t *)cData, len);
            break;
        case CastType::Int322Int16:
            ret = baidu::xpu::api::cast<int, int16_t>(
                context->XPUHandle(), (int *)aData, (int16_t *)cData, len);
            break;
        case CastType::Int162Float:
            ret = baidu::xpu::api::cast<int16_t, float>(
                context->XPUHandle(), (int16_t *)aData, (float *)cData, len);
            break;
        case CastType::Int162Int32:
            ret = baidu::xpu::api::cast<int16_t, int>(
                context->XPUHandle(), (int16_t *)aData, (int *)cData, len);
            break;
        case CastType::Int82Float:
            ret = baidu::xpu::api::cast<int8_t, float>(
                context->XPUHandle(), (int8_t *)aData, (float *)cData, len);
            break;
        case CastType::Int82Int16:
            ret = baidu::xpu::api::cast<int8_t, int16_t>(
                context->XPUHandle(), (int8_t *)aData, (int16_t *)cData, len);
            break;
        case CastType::Int82Int32:
            ret = baidu::xpu::api::cast<int8_t, int>(
                context->XPUHandle(), (int8_t *)aData, (int *)cData, len);
            break;
        case CastType::Int322Int64:
            ret = baidu::xpu::api::cast<int, int64_t>(
                context->XPUHandle(), (int *)aData, (int64_t *)cData, len);
            break;
        case CastType::Int642Int32:
            ret = baidu::xpu::api::cast<int64_t, int>(
                context->XPUHandle(), (int64_t *)aData, (int *)cData, len);
            break;
        case CastType::Int642Float:
            ret = baidu::xpu::api::cast<int64_t, float>(
                context->XPUHandle(), (int64_t *)aData, (float *)cData, len);
            break;
        case CastType::Float162Float:
            ret = baidu::xpu::api::cast<float16, float>(
                context->XPUHandle(), (float16 *)aData, (float *)cData, len);
            break;
        default:
            IT_TODO_HALT();
        }
        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::XPU, OpType::Cast, DataType::Float32, CastXdnn,
                "Cast_xdnn_XPU_Float32");
}; // namespace infini