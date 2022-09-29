#include "operators/gather.h"
#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_runtime.h"
#include "cuda/gather.h"

namespace infini {
class GatherCuda : public CudaKernelWithoutConfig {
    void initGatherMetaData(GatherMetaData &metaData,
                            const Operator &_op) const {
        memset(&metaData, 0, sizeof(metaData));
        auto op = as<GatherObj>(_op);
        auto in = op->getInputs(0);
        auto index = op->getInputs(1);
        auto out = op->getOutput();
        metaData.indexValue = index->getRawDataPtr<int *>();
        metaData.axis = op->getAxis();
        metaData.inNDim = in->getDims().size();
        metaData.outNDim = out->getDims().size();
        metaData.idxNDim = index->getDims().size();
        for (int i = 0; i < metaData.outNDim; ++i)
            metaData.outDim[i] = out->getDims()[i];
        for (int i = 0; i < metaData.idxNDim; ++i) {
            metaData.idxDim[i] = index->getDims()[i];
            metaData.idxStride[i] = index->getStride()[i];
        }
        for (int i = 0; i < metaData.inNDim; ++i) {
            metaData.inStride[i] = in->getStride()[i];
        }
    }

    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {

        auto input = op->getInputs(0);
        auto index = op->getInputs(1);

        GatherMetaData metaData;
        initGatherMetaData(metaData, op);

        auto inData = input->getRawDataPtr<float *>();
        auto outData = op->getOutput()->getRawDataPtr<float *>();
        gather_kernel(inData, outData, metaData, op->getOutput()->size());
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Gather, DataType::Float32, GatherCuda,
                "Gather_CUDA_Float32");
} // namespace infini
