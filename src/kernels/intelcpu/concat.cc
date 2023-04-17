#include "operators/concat.h"
#include "intelcpu/mkl_kernel_without_config.h"
#include "intelcpu/mkl_runtime.h"

namespace infini {
class MklConcat : public MklKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConcatObj>(_op);
        auto context = dynamic_cast<const MklRuntimeObj *>(_context);

        //  create user memory that describes data layout in the buffers
        std::vector<dnnl::memory::desc> srcsMd;
        std::vector<dnnl::memory> srcs;

        for (size_t i = 0; i < op->getInputs().size(); i++) {
            std::vector<dnnl_dim_t> dims;
            auto inDims = op->getInputs(i)->getDims();
            int ndim = inDims.size();
            for (int j = 0; j < ndim; ++j)
                dims.push_back(inDims.at(j));

            auto md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                         getUserFormatTag(dims.size()));
            srcsMd.push_back(md);

            auto srcMemory =
                dnnl::memory(md, context->getEngine(),
                             op->getInputs(i)->getRawDataPtr<float *>());
            srcs.push_back(srcMemory);
        }

        std::vector<dnnl_dim_t> dims;
        auto oDims = op->getOutput(0)->getDims();
        int ndim = oDims.size();
        for (int i = 0; i < ndim; ++i)
            dims.push_back(oDims.at(i));

        auto dstMd = dnnl::memory::desc(dims, dnnl::memory::data_type::f32,
                                        getUserFormatTag(dims.size()));
        auto primDesc =
            dnnl::concat::primitive_desc(dstMd, static_cast<int>(op->getDim()),
                                         srcsMd, context->getEngine());

        float *const dstData = op->getOutput()->getRawDataPtr<float *>();
        auto output = dnnl::memory(dstMd, context->getEngine(), dstData);

        // create and execute primitive
        std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_DST, output}};
        for (int i = 0; i < (int)srcs.size(); i++) {
            args.insert({DNNL_ARG_MULTIPLE_SRC + i, srcs.at(i)});
        }
        dnnl::concat(primDesc).execute(context->getStream(), args);
    }
};
REGISTER_KERNEL(Device::INTELCPU, OpType::Concat, DataType::Float32, MklConcat,
                "Concat_Mkl_Float32");
}; // namespace infini
