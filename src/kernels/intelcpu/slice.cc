#include "operators/slice.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklSlice : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SliceObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        std::vector<dnnl_dim_t> dims;
        for (size_t i = 0; i < op->getInputs(0)->getDims().size(); ++i)
            dims.push_back(op->getInputs(0)->getDims()[i]);

        // create src md
        auto srcMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));

        // dst md
        auto oDims = op->getOutput(0)->getDims();
        int ndim = oDims.size();
        std::vector<dnnl_dim_t> sDims, offsets;
        for (int i = 0; i < ndim; ++i) {
            sDims.push_back(oDims.at(i));
            offsets.push_back(op->getStarts().at(i));
        }
        auto sliceMd = srcMd.submemory_desc(sDims, offsets);
        auto sliceMemory =
            dnnl::memory(sliceMd, context->getEngine(),
                         op->getInputs(0)->getRawDataPtr<float *>());

        auto dstMd = dnnl::memory::desc(sDims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(sDims.size()));
        auto output = dnnl::memory(dstMd, context->getEngine(),
                                   op->getOutput(0)->getRawDataPtr<float *>());

        // copy data to dst
        dnnl::reorder(sliceMemory, output)
            .execute(context->getStream(),
                     {{DNNL_ARG_FROM, sliceMemory}, {DNNL_ARG_TO, output}});
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Slice, DataType::Float32, MklSlice,
                "Slice_Mkl_Float32");
} // namespace infini
