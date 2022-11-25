#include "operators/batch_norm.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklBatchNorm : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<BatchNormObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        float *const srcData = op->getInputs(0)->getRawDataPtr<float *>();
        float *const dstData = op->getOutput()->getRawDataPtr<float *>();

        //  create user memory that describes data layout in the buffers
        std::vector<dnnl_dim_t> dims;
        for (size_t i = 0; i < op->getInputs(0)->getDims().size(); ++i)
            dims.push_back(op->getInputs(0)->getDims()[i]);

        auto srcMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));
        auto srcMemory = dnnl::memory(srcMd, context->getEngine(), srcData);

        auto dstMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));
        auto output = dnnl::memory(dstMd, context->getEngine(), dstData);

        std::vector<dnnl_dim_t> meanDims(op->getInputs(0)->getDims().size(), 1);
        meanDims[1] = op->getInputs(0)->getDims()[1];
        auto meanMd = dnnl::memory::desc(meanDims, dnnl::memory::data_type::f32,
                                         getUserFormatTag(meanDims.size()));

        auto meanMemory =
            dnnl::memory(meanMd, context->getEngine(),
                         op->getInputs(1)->getRawDataPtr<float *>());
        auto varMemory =
            dnnl::memory(meanMd, context->getEngine(),
                         op->getInputs(2)->getRawDataPtr<float *>());
        auto scaleMemory =
            dnnl::memory(meanMd, context->getEngine(),
                         op->getInputs(3)->getRawDataPtr<float *>());
        auto baisMemory =
            dnnl::memory(meanMd, context->getEngine(),
                         op->getInputs(4)->getRawDataPtr<float *>());
        using op_desc_t = dnnl::batch_normalization_forward::desc;
        using pd_t = dnnl::batch_normalization_forward::primitive_desc;

        // use_global_stats stands for use mean and var as inputs
        auto opDesc =
            op_desc_t(dnnl::prop_kind::forward_inference, srcMd, op->getEps(),
                      dnnl::normalization_flags::use_global_stats |
                          dnnl::normalization_flags::use_shift |
                          dnnl::normalization_flags::use_scale);
        auto primDesc = pd_t(opDesc, context->getEngine());

        // create and execute primitive
        dnnl::batch_normalization_forward(primDesc).execute(
            context->getStream(), {{DNNL_ARG_SRC, srcMemory},
                                   {DNNL_ARG_DST, output},
                                   {DNNL_ARG_MEAN, meanMemory},
                                   {DNNL_ARG_VARIANCE, varMemory},
                                   {DNNL_ARG_SCALE, scaleMemory},
                                   {DNNL_ARG_SHIFT, baisMemory}});
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::BatchNorm, DataType::Float32,
                MklBatchNorm, "BatchNorm_Mkl_Float32");
}; // namespace infini
