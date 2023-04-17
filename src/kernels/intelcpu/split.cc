#include "operators/split.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklSplit : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<SplitObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        std::vector<dnnl_dim_t> dims;
        for (size_t i = 0; i < op->getInputs(0)->getDims().size(); ++i)
            dims.push_back(op->getInputs(0)->getDims()[i]);

        // create src md
        auto srcMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));

        // dst md
        std::vector<dnnl::memory::desc> dstsMd;
        std::vector<dnnl::memory> dsts;
        int offset = 0;
        for (size_t i = 0; i < op->getOutputs().size(); i++) {
            auto oDims = op->getOutput(i)->getDims();
            int ndim = oDims.size();
            std::vector<dnnl_dim_t> dims, offsets(ndim, 0);
            for (int i = 0; i < ndim; ++i) {
                dims.push_back(oDims.at(i));
            }
            offsets[op->getDim()] = offset;
            auto splitMd = srcMd.submemory_desc(dims, offsets);
            auto splitMemory =
                dnnl::memory(splitMd, context->getEngine(),
                             op->getInputs(0)->getRawDataPtr<float *>());

            auto dstMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                            getUserFormatTag(dims.size()));
            auto output =
                dnnl::memory(dstMd, context->getEngine(),
                             op->getOutput(i)->getRawDataPtr<float *>());

            // copy data to dst
            dnnl::reorder(splitMemory, output)
                .execute(context->getStream(),
                         {{DNNL_ARG_FROM, splitMemory}, {DNNL_ARG_TO, output}});

            offset += dims.at(op->getDim());
        }
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Split, DataType::Float32, MklSplit,
                "Split_Mkl_Float32");
}; // namespace infini
