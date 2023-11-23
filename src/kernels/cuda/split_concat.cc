#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_split_concat.h"
#include "operators/concat.h"
#include "operators/split.h"
#include <functional>

namespace infini {

class CudaCompute {
    void initComposedTensorMetadata(ComposedTensorMetadata &metadata,
                                    Tensor tensor) const {
        int nDims = tensor->getRank();
        auto strides = tensor->getStride();
        IT_ASSERT(strides.size() == (size_t)nDims);
        for (int i = 0; i < nDims; ++i) {
            metadata.dimSize[i] = tensor->getDims().at(i);
            metadata.stride[i] = strides.at(i);
        }
        metadata.data = tensor->getRawDataPtr<float *>();
    }

    void initElementTensorMetadata(ElementTensorMetadata &metadata,
                                   TensorVec tensors, int idx, int dim,
                                   int &dimBgIdx, int &batchCounter) const {
        int nTensors = tensors.size();
        for (; batchCounter < BATCH_SIZE && idx + batchCounter < nTensors;
             ++batchCounter) {
            auto tensor = tensors.at(idx + batchCounter);
            auto dimSize = tensor->getDims()[dim];
            metadata.data[batchCounter] = tensor->getRawDataPtr<float *>();
            metadata.dimBgNo[batchCounter] = dimBgIdx;
            metadata.dimSize[batchCounter] = dimSize;
            metadata.nElements[batchCounter] = tensor->size();
            dimBgIdx += dimSize;
        }
    }

  public:
    void do_compute(Tensor composedTensor, TensorVec elementsTensor, int dim,
                    int nDims, bool isSplit) const {
        IT_ASSERT(nDims <= DIM_MAX_SIZE);

        ComposedTensorMetadata composedMetadata;
        initComposedTensorMetadata(composedMetadata, composedTensor);

        int dimBgNo = 0;
        int nElemets = elementsTensor.size();
        for (int i = 0; i < nElemets; i += BATCH_SIZE) {
            ElementTensorMetadata elemMetadata;
            int batchCounter = 0;
            initElementTensorMetadata(elemMetadata, elementsTensor, i, dim,
                                      dimBgNo, batchCounter);
            split_concat_kernel(elemMetadata, composedMetadata, dim,
                                batchCounter, nDims, isSplit);
        }
    }
};

class ConcatCuda : private CudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto inputs = _op->getInputs();
        if (inputs.size() == 2) {
            for (size_t i = 0; i < 2; i++) {
                if (inputs[i]->size() == 0) {
                    auto inData =
                        _op->getInputs(1 - i)->getRawDataPtr<void *>();
                    auto outData =
                        _op->getOutputs()[0]->getRawDataPtr<void *>();
                    cudaMemcpyAsync(outData, inData,
                                    _op->getInputs(1 - i)->getBytes(),
                                    cudaMemcpyDeviceToDevice);
                    return;
                }
            }
        }
        do_compute(_op->getOutput(), _op->getInputs(),
                   as<ConcatObj>(_op)->getDim(), _op->getOutput()->getRank(),
                   false);
    }
};

class SplitCuda : private CudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        do_compute(_op->getInputs(0), _op->getOutputs(),
                   as<SplitObj>(_op)->getDim(), _op->getInputs(0)->getRank(),
                   true);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Concat, DataType::Float32, ConcatCuda,
                "Concat_CUDA_Float32");
REGISTER_KERNEL(Device::CUDA, OpType::Split, DataType::Float32, SplitCuda,
                "Split_CUDA_Float32");
} // namespace infini
