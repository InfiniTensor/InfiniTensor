#include "operators/resize.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklResize : public MklKernelWithoutConfig {
    dnnl::algorithm getAlgorithm(Ref<ResizeObj> op) const {
        switch (op->getMode()) {
        case ResizeObj::ECoeffMode::nearest: {
            if (op->getNearestMode() !=
                enum_to_underlying(ResizeObj::ENearestMode::floor))
                IT_TODO_HALT();
            return dnnl::algorithm::resampling_nearest;
        }
        case ResizeObj::ECoeffMode::linear:
            return dnnl::algorithm::resampling_linear;

        default:
            IT_TODO_HALT();
        }
        return dnnl::algorithm::resampling_nearest;
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ResizeObj>(_op);

        // only support default coordinate transmode??
        if (op->getCoordinateTransMode() !=
            enum_to_underlying(ResizeObj::ECoordinateTransMode::halfPixel))
            IT_TODO_HALT();

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

        using op_desc_t = dnnl::resampling_forward::desc;
        using pd_t = dnnl::resampling_forward::primitive_desc;

        auto opDesc = op_desc_t(dnnl::prop_kind::forward_inference,
                                getAlgorithm(op), op->getScales(), srcMd);
        auto primDesc = pd_t(opDesc, context->getEngine());

        // create and execute primitive
        dnnl::resampling_forward(primDesc).execute(
            context->getStream(),
            {{DNNL_ARG_SRC, srcMemory}, {DNNL_ARG_DST, output}});
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Resize, DataType::Float32, MklResize,
                "Resize_Mkl_Float32");
}; // namespace infini
