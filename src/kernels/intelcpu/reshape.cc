#include "operators/reshape.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklReshape : public MklKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {

        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        std::vector<dnnl_dim_t> dims;
        for (size_t i = 0; i < op->getInputs(0)->getDims().size(); ++i)
            dims.push_back(op->getInputs(0)->getDims()[i]);

        // create src md and src memory
        auto srcMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));

        // dst md
        auto oDims = op->getOutput(0)->getDims();
        int ndim = oDims.size();
        std::vector<dnnl_dim_t> reshapeDims;
        for (int i = 0; i < ndim; ++i) {
            reshapeDims.push_back(oDims.at(i));
        }
        auto reshapeMd = srcMd.reshape(reshapeDims);
        auto reshapeMemory =
            dnnl::memory(reshapeMd, context->getEngine(),
                         op->getInputs(0)->getRawDataPtr<float *>());

        auto dstMd =
            dnnl::memory::desc(reshapeDims, dnnl::memory::data_type::f32,
                               getUserFormatTag(reshapeDims.size()));
        auto output = dnnl::memory(dstMd, context->getEngine(),
                                   op->getOutput(0)->getRawDataPtr<float *>());

        // copy data to dst
        dnnl::reorder(reshapeMemory, output)
            .execute(context->getStream(),
                     {{DNNL_ARG_FROM, reshapeMemory}, {DNNL_ARG_TO, output}});
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Reshape, DataType::Float32,
                MklReshape, "Reshape_Mkl_Float32");
REGISTER_KERNEL(Device::INTELCPU, OpType::Identity, DataType::Float32,
                MklReshape, "Identify_Mkl_Float32");
REGISTER_KERNEL(Device::INTELCPU, OpType::Flatten, DataType::Float32,
                MklReshape, "Flatten_Mkl_Float32");
}; // namespace infini
