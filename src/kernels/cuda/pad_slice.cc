#include "cuda/cuda_kernel_wihtout_config.h"
#include "cuda/cuda_pad_slice.h"
#include "operators/pad.h"
#include "operators/slice.h"
namespace infini {
class PadSliceCudaCompute {
  public:
    void do_compute(Tensor partTensor, Tensor wholeTensor, const Shape &begNos,
                    bool isPad) const {
        int nDims = partTensor->getRank();
        IT_ASSERT(MAX_DIM >= nDims);
        TransMetaData metadata;
        for (int i = 0; i < nDims; i++) {
            metadata.begNum[i] = begNos[i];
            metadata.wholeNDim[i] = wholeTensor->getDims()[i];
            metadata.partNDim[i] = partTensor->getDims()[i];
            metadata.partStride[i] = partTensor->getStride()[i];
        }
        metadata.DType = partTensor->getDType().getIndex();
        pad_slice_kernel(partTensor->getRawDataPtr<void *>(),
                         wholeTensor->getRawDataPtr<void *>(), metadata, nDims,
                         wholeTensor->size(), isPad);
    }
};

class PadCuda : private PadSliceCudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        do_compute(op->getInputs(0), op->getOutput(), as<PadObj>(op)->getPads(),
                   true);
    }
};

class SliceCuda : private PadSliceCudaCompute, public CudaKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        bool one_axis = true;
        auto begNos = as<SliceObj>(op)->getStarts();
        auto input = op->getInputs(0);
        auto output = op->getOutput();
        int ndim = output->getRank();
        auto output_shape = output->getDims();
        auto input_shape = input->getDims();

        int dtype_size = 4;
        if (input->getDType().getIndex() == 1) {
            dtype_size = 4;
        }
        int slice_axis = -1;
        for (int i = 0; i < ndim; ++i) {
            if (begNos[i] > 0 || output_shape[i] < input_shape[i]) {
                if (slice_axis != -1) {
                    // printf("Only one axis slice is supported\n");
                    one_axis = false;
                }
                slice_axis = i;
            }
        }

        if (slice_axis == -1) {
            printf("No slice axis found\n");
        }

        // 计算 inner/outer size
        size_t inner = 1; // 切片轴之后的元素数量
        for (int i = slice_axis + 1; i < ndim; ++i) {
            inner *= input_shape[i];
        }

        size_t outer = 1; // 切片轴之前的元素数量
        for (int i = 0; i < slice_axis; ++i) {
            outer *= input_shape[i];
        }

        int slice_len = output_shape[slice_axis]; // 要 copy 的段长度
        int input_axis_len = input_shape[slice_axis];
        int start = begNos[slice_axis];

        size_t copy_bytes = slice_len * inner * dtype_size;
        if (one_axis) {
            if (slice_axis == ndim - 1) {
                size_t total_slice_blocks = outer; // 每个 block 是连续的数据
                size_t copy_block_bytes = slice_len * dtype_size;
                const char *src =
                    static_cast<const char *>(input->getRawDataPtr<void *>()) +
                    start * dtype_size;
                char *dst =
                    static_cast<char *>(output->getRawDataPtr<void *>());

                cudaMemcpy2DAsync(
                    dst,              // dst ptr
                    copy_block_bytes, // dst pitch
                    src,              // src ptr
                    input_axis_len *
                        dtype_size,     // src pitch (跳过 input 的行尾)
                    copy_block_bytes,   // 每次拷贝多少列
                    total_slice_blocks, // 拷贝多少行
                    cudaMemcpyDeviceToDevice, CUDAStream::getCurrentStream());
            } else {
                for (size_t i = 0; i < outer; ++i) {
                    const void *src =
                        (const char *)input->getRawDataPtr<void *>() +
                        ((i * input_axis_len + start) * inner) * dtype_size;
                    void *dst = (char *)output->getRawDataPtr<void *>() +
                                (i * slice_len * inner) * dtype_size;

                    cudaMemcpyAsync(dst, src, copy_bytes,
                                    cudaMemcpyDeviceToDevice,
                                    CUDAStream::getCurrentStream());
                }
            }

        } else {
            do_compute(op->getOutput(), op->getInputs(0),
                       as<SliceObj>(op)->getStarts(), false);
        }
        // do_compute(op->getOutput(), op->getInputs(0),
        //            as<SliceObj>(op)->getStarts(), false);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Slice, SliceCuda, "Slice__CUDA");

REGISTER_KERNEL(Device::CUDA, OpType::Pad, PadCuda, "Pad__CUDA");

} // namespace infini
