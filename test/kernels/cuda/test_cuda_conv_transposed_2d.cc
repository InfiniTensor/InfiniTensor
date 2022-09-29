#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

void testConvTransposedCudnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    const auto &[N, C, H, W, F, R, S] = tuple{1, 1, 2, 2, 1, 4, 4};
    const int stride = 1, padding = 0, dilation = 1;
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = CpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({N, F, H, H}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({F, C, R, S}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<ConvTransposed2dObj>(i0Cuda, w0Cuda, nullptr,
                                                  padding, padding, stride,
                                                  stride, dilation, dilation);
    gCuda->dataMalloc();
    // Execute on CUDA
    cuda->run(gCuda);
    // copy output from CUDA to CPU
    auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
    // check results on CPU
    EXPECT_TRUE(o0Cpu->equalData(ansVec));
}

TEST(cuDNN_ConvTransposed, run) {
    testConvTransposedCudnn(IncrementalGenerator(),
                            vector<float>{0.,  0.,  1.,  2.,  3.,  0.,  6.,
                                          12., 18., 16., 8.,  30., 36., 42.,
                                          32., 16., 54., 60., 66., 48., 24.,
                                          62., 67., 72., 45.});
}

TEST(cuDNN_ConvTransposed, tune) {
    Runtime cpu = CpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 448, 2, 2}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({448, 256, 4, 4}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<ConvTransposed2dObj>(i0Cuda, w0Cuda, nullptr);
    // allocate CUDA memory
    gCuda->dataMalloc();
    // Execute on CUDA
    bool tune = true;
    cuda->run(gCuda, tune);
    // print a tensor/operator/graph by print()
    gCuda->print();
    // check record
    auto kernelAttrs =
        KernelAttrs{Device::CUDA, conv->getOpType(), DataType::Float32};
    auto perfKey = PerfEngine::Key{kernelAttrs, conv->getOpPerfKey()};
    std::optional<PerfRecord> perfData =
        PerfEngine::getInstance().getPerfData(perfKey);
    ASSERT_TRUE(perfData.has_value());
}

} // namespace infini
