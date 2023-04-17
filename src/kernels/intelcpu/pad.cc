#include "operators/pad.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklPad : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PadObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        std::vector<dnnl_dim_t> dims;
        for (size_t i = 0; i < op->getInputs(0)->getDims().size(); ++i) {
            dims.push_back(op->getInputs(0)->getDims()[i]);
        }
        auto paddedMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                           getUserFormatTag(dims.size()));

        // dst md
        auto oDims = op->getOutput(0)->getDims();
        int ndim = oDims.size();
        std::vector<dnnl_dim_t> paddedDims, offsets;
        for (int i = 0; i < ndim; ++i) {
            paddedDims.push_back(oDims.at(i));
            paddedMd.data.padded_dims[i] = oDims.at(i);
            paddedMd.data.padded_offsets[i] = op->getPads().at(i);
            offsets.push_back(op->getPads().at(i));
        }
        // will fill padded area with zero.
        auto paddedMemory =
            dnnl::memory(paddedMd, context->getEngine(),
                         op->getOutput(0)->getRawDataPtr<float *>());

        auto dstMd =
            dnnl::memory::desc(paddedDims, dnnl::memory::data_type::f32,
                               getUserFormatTag(paddedDims.size()));

        // copy src to the submemory of dst
        // create submemory
        auto md = dstMd.submemory_desc(dims, offsets);
        auto mem = dnnl::memory(md, context->getEngine(),
                                op->getOutput(0)->getRawDataPtr<float *>());

        auto srcMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));
        auto srcMemory =
            dnnl::memory(srcMd, context->getEngine(),
                         op->getInputs(0)->getRawDataPtr<float *>());

        // copy data to submemory
        dnnl::reorder(srcMemory, mem)
            .execute(context->getStream(),
                     {{DNNL_ARG_FROM, srcMemory}, {DNNL_ARG_TO, mem}});
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Pad, DataType::Float32, MklPad,
                "Pad_Mkl_Float32");
} // namespace infini
