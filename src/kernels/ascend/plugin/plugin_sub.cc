#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_plugin_sub_kernel.h"
#include "ascend/ascend_runtime.h"
#include "operators/ascend_plugin_sub.h"

namespace infini {

class PluginSubKernelAscend : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AscendPluginSubObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        auto input_shape = op->getInputs(0)->getDims();
        auto output_shape = op->getOutput(0)->getDims();
        auto input_size = op->getInputs(0)->size();
        auto output_size = op->getOutput(0)->size();

        PluginMetaData plugin_meta_data = {
            input_shape,
            output_shape,
            input_size,
            output_size,
            // DataType::Float16,
            5,
            1,
        };
        aclrtStream stream = context->getStream();
        plugin_sub_kernel(op->getInputs(0)->getRawDataPtr<float *>(),
                          op->getOutput(0)->getRawDataPtr<float *>(),
                          plugin_meta_data, (void *)stream);
        // PluginSub<<<8, nullptr, context->getStream()>>>(
        //     op->getInputs(0)->getRawDataPtr<float *>(),
        //     op->getOutput(0)->getRawDataPtr<float *>(), inputSize,
        //     outputSize, C);
        aclrtSynchronizeStream(stream);
        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::AscendPluginSub, PluginSubKernelAscend,
                "PluginSubKernelAscend");
} // namespace infini
