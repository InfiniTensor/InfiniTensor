#include "ascend/ascend_kernel_without_config.h"
#include "ascend/ascend_plugin_sub_kernel.h"
#include "ascend/ascend_runtime.h"
#include "operators/ascend_plugin_sub.h"

extern "C" void plugin_sub_kernel(void *in, void *out, PluginMetaData metaData,
                                  void *stream, int dtype);

namespace infini {

class PluginSubKernelAscend : public ASCENDKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<AscendPluginSubObj>(_op);
        auto context = dynamic_cast<const ASCENDRuntimeObj *>(_context);
        auto input = op->getInputs(0)->getRawDataPtr<void *>();
        auto output = op->getOutput(0)->getRawDataPtr<void *>();
        auto input_shape = op->getInputs(0)->getDims();
        auto output_shape = op->getOutput(0)->getDims();
        auto input_size = op->getInputs(0)->size();
        auto output_size = op->getOutput(0)->size();

        PluginMetaData plugin_meta_data = {
            input_shape, output_shape, input_size, output_size, 5, 1,
        };
        aclrtStream stream = context->getStream();

        if (op->getDType() == DataType::Float32) {
            plugin_sub_kernel(input, output, plugin_meta_data, (void *)stream,
                              0);
        } else if (op->getDType() == DataType::Float16) {
            plugin_sub_kernel(input, output, plugin_meta_data, (void *)stream,
                              1);
        } else {
            IT_ASSERT(false, "Unsupported data type");
        }
        aclrtSynchronizeStream(stream);
        return;
    }
};

REGISTER_KERNEL(Device::ASCEND, OpType::AscendPluginSub, PluginSubKernelAscend,
                "PluginSubKernelAscend");
} // namespace infini
