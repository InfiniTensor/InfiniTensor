#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"
#include "operators/matmul.h"
#include "utils/data_generator.h"

namespace infini {
namespace opTimer {

double getPerfConvCudnn(int n, int c, int h, int w, int f, int r, int s,
                        int padh, int padw, int strideh, int stridew,
                        int dilationh, int dilationw, int group,
                        const char *name) {
    // const auto &[n, c, h, w, f, r, s, padh, padw, strideh, stridew,
    // dilationh, dilationw, group] =
    //     tuple{1, 512, 14, 14, 512, 3, 3, 2, 2, 1, 1, 2, 2, 1};
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    IT_ASSERT(c % group == 0);
    Tensor i0Cpu = gCpu->addTensor({n, c, h, w}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f, c / group, r, s}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv =
        gCuda->addOp<ConvObj>(i0Cuda, w0Cuda, nullptr, padh, padw, nullptr,
                              strideh, stridew, dilationh, dilationw);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    bool tune = true;
    cuda->run(gCuda, tune);
    return cuda->getPerfTime(gCuda);
}

double getPerfConvTransposed2dCudnn(int n, int c, int h, int w, int f, int r,
                                    int s, int padh, int padw, int strideh,
                                    int stridew, int dilationh, int dilationw,
                                    int oph, int opw, int group) {
    // const auto &[n, c, h, w, f, r, s, padh, padw, strideh, stridew,
    // dilationh, dilationw, group] =
    //     tuple{1, 512, 14, 14, 512, 3, 3, 2, 2, 1, 1, 2, 2, 1};
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    IT_ASSERT(c % group == 0);
    Tensor i0Cpu = gCpu->addTensor({n, f, h, w}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({f, c / group, r, s}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<ConvTransposed2dObj>(
        i0Cuda, w0Cuda, nullptr, padh, padw, strideh, stridew, dilationh,
        dilationw, oph, opw, group);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    bool tune = true;
    cuda->run(gCuda, tune);
    return cuda->getPerfTime(gCuda);
}

double getPerfMatmulCublas(int b, int m, int n, int k, const char *name) {
    // const auto &[n, c, h, w, f, r, s, padh, padw, strideh, stridew,
    // dilationh, dilationw, group] =
    //     tuple{1, 512, 14, 14, 512, 3, 3, 2, 2, 1, 1, 2, 2, 1};
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({b, m, k}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({b, k, n}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<MatmulObj>(i0Cuda, w0Cuda, nullptr);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    bool tune = true;
    cuda->run(gCuda, tune);
    return cuda->getPerfTime(gCuda);
}

} // namespace opTimer
} // namespace infini
