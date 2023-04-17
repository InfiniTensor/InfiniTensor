#include "operators/pooling.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklPooling : public MklKernelWithoutConfig {
    virtual dnnl::algorithm getAlgorithm() const = 0;

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<PoolingObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        float *const srcData = op->getInputs(0)->getRawDataPtr<float *>();
        float *const dstData = op->getOutput()->getRawDataPtr<float *>();

        //  create user memory that describes data layout in the buffers
        auto [n, c, h, w, r, s] = op->getNCHWRS();
        auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        auto nDim = op->getOutput()->getDims().size();
        auto oh = op->getOutput()->getDims()[nDim - 2];
        auto ow = op->getOutput()->getDims()[nDim - 1];

        auto srcMd = dnnl::memory::desc(
            {n, c, h, w}, dnnl::memory::data_type::f32, getUserFormatTag(nDim));
        auto srcMemory = dnnl::memory(srcMd, context->getEngine(), srcData);

        auto userDstMd =
            dnnl::memory::desc({n, c, oh, ow}, dnnl::memory::data_type::f32,
                               getUserFormatTag(nDim));

        auto dstMd =
            dnnl::memory::desc({n, c, oh, ow}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::any);

        using op_desc_t = dnnl::pooling_v2_forward::desc;
        using pd_t = dnnl::pooling_v2_forward::primitive_desc;

        auto opDesc = op_desc_t(dnnl::prop_kind::forward_inference,
                                getAlgorithm(), srcMd, dstMd, {sh, sw}, {r, s},
                                {dh - 1, dw - 1}, {ph, pw}, {ph, pw});
        auto primDesc = pd_t(opDesc, context->getEngine());

        if (primDesc.dst_desc() == userDstMd) {
            auto output = dnnl::memory(primDesc.dst_desc(),
                                       context->getEngine(), dstData);

            dnnl::pooling_v2_forward(primDesc).execute(
                context->getStream(),
                {{DNNL_ARG_SRC, srcMemory}, {DNNL_ARG_DST, output}});
        } else {
            auto dstMemory =
                dnnl::memory(primDesc.dst_desc(), context->getEngine());

            dnnl::pooling_v2_forward(primDesc).execute(
                context->getStream(),
                {{DNNL_ARG_SRC, srcMemory}, {DNNL_ARG_DST, dstMemory}});

            auto output =
                dnnl::memory(userDstMd, context->getEngine(), dstData);
            dnnl::reorder(dstMemory, output)
                .execute(context->getStream(),
                         {{DNNL_ARG_FROM, dstMemory}, {DNNL_ARG_TO, output}});
        }
    }
};

class MklAvgPool : public MklPooling {
    dnnl::algorithm getAlgorithm() const override {
        return dnnl::algorithm::pooling_avg_include_padding;
    }
};

class MklMaxPool : public MklPooling {
    dnnl::algorithm getAlgorithm() const override {
        return dnnl::algorithm::pooling_max;
    }
};

REGISTER_KERNEL(Device::INTELCPU, OpType::AvgPool, DataType::Float32,
                MklAvgPool, "AvgPool_Mkl_Float32");
REGISTER_KERNEL(Device::INTELCPU, OpType::MaxPool, DataType::Float32,
                MklMaxPool, "MaxPool_Mkl_Float32");
} // namespace infini
