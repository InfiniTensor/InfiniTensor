#include "operators/where.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

namespace infini {
class WhereXdnn : public KUNLUNKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<WhereObj>(_op);
        auto context = dynamic_cast<const KUNLUNRuntimeObj *>(_context);

        void *const aData =
            (op->getInputs(0)->getRawDataPtr<void *>()); // inputX
        void *const bData =
            (op->getInputs(1)->getRawDataPtr<void *>()); // inputY
        void *const cData =
            (op->getInputs(2)->getRawDataPtr<void *>()); // condition
        void *const dData =
            (op->getOutput()->getRawDataPtr<void *>()); // output

        auto aDim = op->getInputs(0)->getDims(); // dimX
        auto bDim = op->getInputs(1)->getDims(); // dimY
        auto cDim = op->getInputs(2)->getDims(); // dimCondition
        auto dDim = op->getOutput()->getDims();  //  dimOutput

        auto numAData = op->getInputs(1)->size();
        auto numCData = op->getInputs(2)->size();

        int ret;

        void *wkspace = context->getWorkspace(numCData * sizeof(bool));
        ret = baidu::xpu::api::cast<int8_t, bool>(context->KUNLUNHandle(),
                                                  (int8_t *)cData,
                                                  (bool *)wkspace, numCData);

        void *broadcastWkspace = static_cast<void *>(
            (bool *)context->getWorkspace(numAData * sizeof(float)) +
            numCData * sizeof(bool));
        ret = baidu::xpu::api::broadcast<float>(
            context->KUNLUNHandle(), (float *)bData, (float *)broadcastWkspace,
            bDim, aDim);

        ret = baidu::xpu::api::select<float>(
            context->KUNLUNHandle(), (bool *)wkspace, (float *)aData,
            (float *)broadcastWkspace, (float *)dData, cDim, aDim);

        assert(ret == 0);
        return;
    }
};

REGISTER_KERNEL(Device::KUNLUN, OpType::Where, DataType::Float32, WhereXdnn,
                "Where_xdnn_KUNLUN_Float32");
}; // namespace infini
