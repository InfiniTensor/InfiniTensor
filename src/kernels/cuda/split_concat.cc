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
                                    cudaMemcpyDeviceToDevice,
                                    CUDAStream::getCurrentStream());
                    return;
                }
            }
        }
        int num_inputs = inputs.size();
        auto output = _op->getOutput();
        int ndim = output->getRank();
        int axis = as<ConcatObj>(_op)->getDim();
        auto output_shape = output->getDims();
        int dtype_size = _op->getDType().getSize();

        // std::cout << "concat" << std::endl;
        // for (int i = 0; i < ndim; i++) {
        //     printf("%d ", output_shape[i]);
        // }
        // printf("\n");
        // printf("[ndim, axis]:[%d, %d]\n", ndim, axis);
        // for (int n = 0; n < num_inputs; n++) {
        //     for (int i = 0; i < ndim; i++) {
        //         printf("%d ", inputs[n]->getDims()[i]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        size_t inner_block = 1;
        for (int i = axis + 1; i < ndim; ++i)
            inner_block *= output_shape[i];

        size_t outer_block = 1;
        for (int i = 0; i < axis; ++i)
            outer_block *= output_shape[i];
        size_t offset_axis = 0;
        if (axis == ndim - 1) {
            for (int i = 0; i < num_inputs; ++i) {
                size_t axis_len = _op->getInputs(i)->getDims()[axis];

                // ✅ 优化点：直接拷贝 outer_block 行的连续 block
                size_t copy_elems = outer_block * axis_len * inner_block;
                size_t copy_bytes = copy_elems * dtype_size;

                void *dst = (char *)output->getRawDataPtr<void *>() +
                            (offset_axis * inner_block * dtype_size);
                void *src = (void *)inputs[i]->getRawDataPtr<void *>();

                cudaMemcpyAsync(dst, src, copy_bytes, cudaMemcpyDeviceToDevice,
                                CUDAStream::getCurrentStream());

                offset_axis += axis_len;
            }
        } else {
            for (int i = 0; i < num_inputs; ++i) {
                size_t axis_len = _op->getInputs(i)->getDims()[axis];

                size_t copy_bytes = axis_len * inner_block * dtype_size;
                for (size_t j = 0; j < outer_block; ++j) {
                    void *dst = (char *)output->getRawDataPtr<void *>() +
                                ((j * output_shape[axis] + offset_axis) *
                                 inner_block * dtype_size);
                    void *src = (char *)inputs[i]->getRawDataPtr<void *>() +
                                (j * axis_len * inner_block * dtype_size);

                    cudaMemcpyAsync(dst, src, copy_bytes,
                                    cudaMemcpyDeviceToDevice,
                                    CUDAStream::getCurrentStream());
                }

                offset_axis += axis_len;
            }
        }

        // if (_op->getDType() == DataType::Float32) {
        //     do_compute<float>(_op->getOutput(), _op->getInputs(),
        //                       as<ConcatObj>(_op)->getDim(),
        //                       _op->getOutput()->getRank(), false);
        // } else if (_op->getDType() == DataType::Float16) {
        //     do_compute<half>(_op->getOutput(), _op->getInputs(),
        //                      as<ConcatObj>(_op)->getDim(),
        //                      _op->getOutput()->getRank(), false);
        // } else {
        //     IT_ASSERT(false);
        // }
    }
};

class SplitCuda : private CudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        int axis = as<SplitObj>(_op)->getDim();
        int ndim = _op->getInputs(0)->getRank();

        auto input = _op->getInputs(0);
        auto outputs = _op->getOutputs();
        int num_outputs = outputs.size();
        auto input_shape = input->getDims();
        int dtype_size = 2;
        if (_op->getDType() == DataType::Float32) {
            dtype_size = 4;
        }
        int outer_count = 1;
        for (int i = 0; i < axis; ++i) {
            outer_count *= input_shape[i];
        }
        std::vector<int> split_sizes;
        for (int i = 0; i < num_outputs; ++i) {
            split_sizes.push_back(
                outputs[i] != nullptr ? outputs[i]->getDims()[axis] : 0);
        }
        if (axis == ndim - 1) {
            // 最后一维切分，可直接整块 memcpy
            size_t axis_offset = 0;

            for (int i = 0; i < num_outputs; ++i) {
                int Bi = split_sizes[i];

                // 合并所有 outer_count 行后的总元素数（Bi 元素 × outer_count
                // 批）
                size_t copy_elems = outer_count * Bi;
                size_t copy_bytes = copy_elems * dtype_size;

                // 输入偏移：从 input 的 axis_offset 开始（每个元素按 element
                // 计）
                size_t input_offset = axis_offset * dtype_size;

                // 直接获取 raw pointer
                void *dst = outputs[i]->getRawDataPtr<void *>();
                const void *src =
                    (const char *)input->getRawDataPtr<void *>() + input_offset;

                // 一次性拷贝 outer_count 行的 Bi 元素
                cudaMemcpyAsync(dst, src, copy_bytes, cudaMemcpyDeviceToDevice,
                                CUDAStream::getCurrentStream());

                axis_offset += Bi;
            }
        } else {
            int inner_stride = 1;
            for (int i = axis + 1; i < ndim; ++i) {
                inner_stride *= input_shape[i];
            }

            // 每个块的元素数量
            int block_size = inner_stride * dtype_size;

            // 当前 offset（以 element 为单位，不是 byte）
            int axis_offset = 0;

            for (int i = 0; i < num_outputs; ++i) {
                int Bi = split_sizes[i];

                for (int j = 0; j < outer_count; ++j) {
                    size_t input_offset =
                        ((j * input_shape[axis] + axis_offset) * inner_stride) *
                        dtype_size;
                    size_t output_offset = (j * Bi * inner_stride) * dtype_size;
                    void *dst = (char *)outputs[i]->getRawDataPtr<void *>() +
                                output_offset;
                    void *src =
                        (char *)input->getRawDataPtr<void *>() + input_offset;
                    cudaMemcpyAsync(dst, src, Bi * block_size,
                                    cudaMemcpyDeviceToDevice,
                                    CUDAStream::getCurrentStream());
                }

                axis_offset += Bi;
            }
        }
        // std::cout << "split" << std::endl;
        // for (int i = 0; i < ndim; i++) {
        //     printf("%d ", input_shape[i]);
        // }
        // printf("\n");
        // printf("[ndim, axis]:[%d, %d]\n", ndim, axis);
        // for (int n = 0; n < num_outputs; n++) {
        //     for (int i = 0; i < ndim; i++) {
        //         printf("%d ", outputs[n]->getDims()[i]);
        //     }
        //     printf("\n");
        // }
        // printf("\n");
        // if (_op->getDType() == DataType::Float32) {
        //     do_compute<float>(_op->getInputs(0), _op->getOutputs(),
        //                       as<SplitObj>(_op)->getDim(),
        //                       _op->getInputs(0)->getRank(), true);
        // } else if (_op->getDType() == DataType::Float16) {
        //     do_compute<half>(_op->getInputs(0), _op->getOutputs(),
        //                      as<SplitObj>(_op)->getDim(),
        //                      _op->getInputs(0)->getRank(), true);
        // } else {
        //     IT_ASSERT(false);
        // }
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Concat, ConcatCuda, "Concat_CUDA");
REGISTER_KERNEL(Device::CUDA, OpType::Split, SplitCuda, "Split_CUDA");

} // namespace infini
