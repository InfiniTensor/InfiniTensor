#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/reduce_mean.h"

namespace infini {
class MklReduce : public MklKernelWithoutConfig {
    dnnl::algorithm getAlgorithm() const {
        return dnnl::algorithm::reduction_mean;
    }

    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ReduceMeanObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        float *const srcData = op->getInputs(0)->getRawDataPtr<float *>();
        float *const dstData = op->getOutput()->getRawDataPtr<float *>();

        //  create user memory that describes data layout in the buffers
        std::vector<dnnl_dim_t> inDims, inStrides;
        for (size_t i = 0; i < op->getInputs(0)->getDims().size(); ++i) {
            inDims.push_back(op->getInputs(0)->getDims()[i]);
            inStrides.push_back(op->getInputs(0)->getStride()[i]);
        }

        std::vector<dnnl_dim_t> oDims(op->getInputs(0)->getDims().size(), 0),
            oStrides(op->getInputs(0)->getDims().size(), 1);
        if (!op->getKeepDims()) {
            oDims = inDims;
            for (size_t i = 0; i < op->getInputs(0)->getDims().size(); ++i) {
                if (op->isReduced(i)) {
                    oDims[i] = 1;
                }
            }
            int stride = 1;
            for (int i = (int)oDims.size() - 1; i >= 0; --i) {
                oStrides[i] = stride;
                stride *= oDims[i];
            }
        } else {
            for (size_t i = 0; i < op->getOutput(0)->getDims().size(); ++i) {
                oDims[i] = op->getOutput(0)->getDims()[i];
                oStrides[i] = op->getOutput(0)->getStride()[i];
            }
        }

        auto srcMd =
            dnnl::memory::desc(inDims, dnnl::memory::data_type::f32, inStrides);
        auto srcMemory = dnnl::memory(srcMd, context->getEngine(), srcData);

        auto dstMd =
            dnnl::memory::desc(oDims, dnnl::memory::data_type::f32, oStrides);
        auto output = dnnl::memory(dstMd, context->getEngine(), dstData);

        using op_desc_t = dnnl::reduction::desc;
        using pd_t = dnnl::reduction::primitive_desc;

        auto opDesc = op_desc_t(getAlgorithm(), srcMd, dstMd, 0, 0);
        auto primDesc = pd_t(opDesc, context->getEngine());

        // create and execute primitive
        dnnl::reduction(primDesc).execute(
            context->getStream(),
            {{DNNL_ARG_SRC, srcMemory}, {DNNL_ARG_DST, output}});
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::ReduceMean, DataType::Float32,
                MklReduce, "ReduceMean_Mkl_Float32");
}; // namespace infini
