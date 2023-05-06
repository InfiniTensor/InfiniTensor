#include "operators/transpose.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklTranspose : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
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
        std::vector<dnnl_dim_t> dstDims;
        Shape permute = op->getPermute(), ori_permute = op->getPermute();
        for (int i = 0; i < ndim; ++i) {
            dstDims.push_back(oDims.at(i));
            permute[i] = std::find(ori_permute.begin(), ori_permute.end(), i) -
                         ori_permute.begin();
        }

        auto permuteMd = srcMd.permute_axes(permute);
        auto permuteMemory =
            dnnl::memory(permuteMd, context->getEngine(),
                         op->getInputs(0)->getRawDataPtr<float *>());

        auto userDstMd =
            dnnl::memory::desc(dstDims, dnnl::memory::data_type::f32,
                               getUserFormatTag(dstDims.size()));
        auto output = dnnl::memory(userDstMd, context->getEngine(),
                                   op->getOutput(0)->getRawDataPtr<float *>());

        // Create memory for output
        if (permuteMd == userDstMd) {
            auto output =
                dnnl::memory(userDstMd, context->getEngine(),
                             op->getOutput(0)->getRawDataPtr<float *>());

            // copy data to dst
            dnnl::reorder(permuteMemory, output)
                .execute(context->getStream(), {{DNNL_ARG_FROM, permuteMemory},
                                                {DNNL_ARG_TO, output}});
        } else {
            auto dstMemory = dnnl::memory(permuteMd, context->getEngine());
            // copy data to dst
            dnnl::reorder(permuteMemory, dstMemory)
                .execute(context->getStream(), {{DNNL_ARG_FROM, permuteMemory},
                                                {DNNL_ARG_TO, dstMemory}});

            auto output =
                dnnl::memory(userDstMd, context->getEngine(),
                             op->getOutput(0)->getRawDataPtr<float *>());
            dnnl::reorder(dstMemory, output)
                .execute(context->getStream(),
                         {{DNNL_ARG_FROM, dstMemory}, {DNNL_ARG_TO, output}});
        }
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Transpose, DataType::Float32,
                MklTranspose, "Transpose_Mkl_Float32");
}; // namespace infini
