#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_split_concat.h"
#include "operators/concat.h"
#include "operators/split.h"
#include <functional>

namespace infini {

class CudaCompute {
    template <typename T>
    void initComposedTensorMetadata(ComposedTensorMetadata<T> &metadata,
                                    Tensor tensor) const {
        int nDims = tensor->getRank();
        auto strides = tensor->getStride();
        IT_ASSERT(strides.size() == (size_t)nDims);
        for (int i = 0; i < nDims; ++i) {
            metadata.dimSize[i] = tensor->getDims().at(i);
            metadata.stride[i] = strides.at(i);
        }
        metadata.data = tensor->getRawDataPtr<T *>();
    }
    template <typename T>
    void initElementTensorMetadata(ElementTensorMetadata<T> &metadata,
                                   TensorVec tensors, int idx, int dim,
                                   int &dimBgIdx, int &batchCounter) const {
        int nTensors = tensors.size();
        for (; batchCounter < BATCH_SIZE && idx + batchCounter < nTensors;
             ++batchCounter) {
            auto tensor = tensors.at(idx + batchCounter);
            auto dimSize = tensor->getDims()[dim];
            metadata.data[batchCounter] = tensor->getRawDataPtr<T *>();
            metadata.dimBgNo[batchCounter] = dimBgIdx;
            metadata.dimSize[batchCounter] = dimSize;
            metadata.nElements[batchCounter] = tensor->size();
            dimBgIdx += dimSize;
        }
    }

  public:
    template <typename T>
    void do_compute(Tensor composedTensor, TensorVec elementsTensor, int dim,
                    int nDims, bool isSplit) const {
        IT_ASSERT(nDims <= DIM_MAX_SIZE);
        ComposedTensorMetadata<T> composedMetadata;
        initComposedTensorMetadata<T>(composedMetadata, composedTensor);

        int dimBgNo = 0;
        int nElemets = elementsTensor.size();
        for (int i = 0; i < nElemets; i += BATCH_SIZE) {
            ElementTensorMetadata<T> elemMetadata;
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
        if (_op->getDType() == DataType::Float32) {
            do_compute<float>(_op->getOutput(), _op->getInputs(),
                              as<ConcatObj>(_op)->getDim(),
                              _op->getOutput()->getRank(), false);
        } else if (_op->getDType() == DataType::Float16) {
            do_compute<half>(_op->getOutput(), _op->getInputs(),
                             as<ConcatObj>(_op)->getDim(),
                             _op->getOutput()->getRank(), false);
        }
    }
};

class SplitCuda : private CudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        if (_op->getDType() == DataType::Float32) {
            do_compute<float>(_op->getInputs(0), _op->getOutputs(),
                              as<SplitObj>(_op)->getDim(),
                              _op->getInputs(0)->getRank(), true);
        } else if (_op->getDType() == DataType::Float16) {
            do_compute<half>(_op->getInputs(0), _op->getOutputs(),
                             as<SplitObj>(_op)->getDim(),
                             _op->getInputs(0)->getRank(), true);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Concat, ConcatCuda, "Concat_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Split, SplitCuda, "Split_CUDA");

} // namespace infini
