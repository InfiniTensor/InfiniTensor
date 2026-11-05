#include "cuda/cuda_kernel_without_config.h"
#include "cuda/cuda_split_concat.h"
#include "operators/concat.h"
#include "operators/split.h"

namespace infini {

// Split/Concat 共用元数据构造和启动流程，只通过 isSplit 改变复制方向。
class CudaCompute {
    // 大张量保留真实 stride，供通用索引和密集布局检查使用。
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
    // 从 idx 开始为本次启动收集分段，最多 BATCH_SIZE 个。
    // dimBgIdx 跨启动批次累加，始终表示分段在整个大张量轴上的位置。
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
                    bool isSplit) const {
        int nDims = composedTensor->getRank();
        IT_ASSERT(nDims <= DIM_MAX_SIZE);
        ComposedTensorMetadata<T> composedMetadata;
        initComposedTensorMetadata<T>(composedMetadata, composedTensor);

        int dimBgNo = 0;
        int nElemets = elementsTensor.size();
        // 这里分批的是小张量数量，与输入形状中的模型 batch 维无关。
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
        // 两个输入中有一个为空时，结果就是另一个输入，直接异步拷贝即可。
        if (inputs.size() == 2) {
            for (size_t i = 0; i < 2; i++) {
                if (inputs[i]->size() == 0) {
                    auto inData =
                        _op->getInputs(1 - i)->getRawDataPtr<void *>();
                    auto outData =
                        _op->getOutputs()[0]->getRawDataPtr<void *>();
                    cudaMemcpyAsync(outData, inData,
                                    _op->getInputs(1 - i)->getBytes(),
                                    cudaMemcpyDeviceToDevice,
                                    CUDAStream::getCurrentStream());
                    return;
                }
            }
        }
        // Concat：输出是大张量，输入列表是分段；false 表示分段写入大张量。
        if (_op->getDType() == DataType::Float32) {
            do_compute<float>(_op->getOutput(), _op->getInputs(),
                              as<ConcatObj>(_op)->getDim(), false);
        } else if (_op->getDType() == DataType::Float16) {
            do_compute<half>(_op->getOutput(), _op->getInputs(),
                             as<ConcatObj>(_op)->getDim(), false);
        } else {
            IT_ASSERT(false);
        }
    }
};

class SplitCuda : private CudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        // Split：输入是大张量，输出列表是分段；true 表示从大张量读取。
        if (_op->getDType() == DataType::Float32) {
            do_compute<float>(_op->getInputs(0), _op->getOutputs(),
                              as<SplitObj>(_op)->getDim(), true);
        } else if (_op->getDType() == DataType::Float16) {
            do_compute<half>(_op->getInputs(0), _op->getOutputs(),
                             as<SplitObj>(_op)->getDim(), true);
        } else {
            IT_ASSERT(false);
        }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Concat, ConcatCuda, "Concat_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Split, SplitCuda, "Split_CUDA");

} // namespace infini
