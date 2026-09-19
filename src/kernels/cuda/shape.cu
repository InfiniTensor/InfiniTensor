#include "cuda/cuda_kernel_wihtout_config.h"

namespace infini {

namespace {

/// The most dimensions this kernel will report. A graph asking for more says so
/// rather than writing past the end of what it was given.
constexpr int SHAPE_MAX_RANK = 8;

/// The dimensions, carried into the kernel by value.
///
/// They could instead be copied from a host buffer, but a copy from pageable
/// host memory is not something a stream capture can record safely: the buffer
/// would have to outlive the capture and still hold these numbers when the
/// graph is replayed, which nothing here can promise. Kernel arguments are
/// stored in the graph node at capture, so passing them this way is correct
/// under replay as well as under a plain launch.
struct ShapeDimsPack {
    int64_t data[SHAPE_MAX_RANK];
};

__global__ void _shape_kernel(int64_t *__restrict__ out, ShapeDimsPack dims,
                              int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rank) {
        out[i] = dims.data[i];
    }
}

} // namespace

/// Writes out the dimensions of the input as the int64 list ONNX asks for.
///
/// The dimensions are known once shapes have been inferred, so there is nothing
/// to work out here beyond putting them where the rest of the graph reads them
/// from. On this device that place is in device memory, which is why a kernel
/// runs at all rather than a plain assignment as on the host.
class ShapeCuda : public CudaKernelWithoutConfig {
    void compute(const Operator &op,
                 const RuntimeObj *_context) const override {
        const auto &dims = op->getInputs(0)->getDims();
        const int rank = static_cast<int>(dims.size());
        IT_ASSERT(rank <= SHAPE_MAX_RANK,
                  "this shape has " + std::to_string(rank) +
                      " dimensions, more than this kernel reports");

        ShapeDimsPack pack{};
        for (int i = 0; i < rank; ++i) {
            pack.data[i] = static_cast<int64_t>(dims[i]);
        }

        auto out = op->getOutput()->getRawDataPtr<int64_t *>();
        _shape_kernel<<<1, SHAPE_MAX_RANK, 0, CUDAStream::getCurrentStream()>>>(
            out, pack, rank);
    }
};

REGISTER_KERNEL(Device::CUDA, OpType::Shape, ShapeCuda, "Shape_CUDA");

} // namespace infini
