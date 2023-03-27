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
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
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

void testConvTransposedNHWCCudnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    const auto &[N, C, H, W, F, R, S] = tuple{1, 1, 2, 2, 2, 4, 4};
    const int stride = 1, padding = 0, dilation = 1;
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({N, H, W, F}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({F, R, S, C}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gCuda->addOp<ConvTransposed2dNHWCObj>(
        i0Cuda, w0Cuda, nullptr, padding, padding, stride, stride, dilation,
        dilation);
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

TEST(cuDNN_ConvTransposedNHWC, run) {
    testConvTransposedNHWCCudnn(IncrementalGenerator(),
                                vector<float>{16,  65,  71,  77,  63,  100, 290,
                                              318, 346, 234, 140, 402, 430, 458,
                                              306, 180, 514, 542, 570, 378, 188,
                                              465, 487, 509, 307});
}

TEST(cuDNN_ConvTransposed, run1) {
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime cuda = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cuda);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 2, 3, 3}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({2, 2, 3, 3}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to CUDA
    Tensor i0Cuda = gCuda->cloneTensor(i0Cpu);
    Tensor w0Cuda = gCuda->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv =
        gCuda->addOp<ConvTransposed2dObj>(i0Cuda, w0Cuda, nullptr, 0, 0);
    gCuda->dataMalloc();
    // Execute on CUDA
    cuda->run(gCuda);
    // copy output from CUDA to CPU
    auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
    // check results on CPU
    EXPECT_TRUE(o0Cpu->equalData(vector<float>{
        162, 351,  569,  413,  224,  405,  876,  1417, 1024, 553,
        747, 1611, 2598, 1869, 1005, 639,  1368, 2191, 1564, 835,
        396, 843,  1343, 953,  506,  243,  531,  866,  629,  341,
        621, 1344, 2173, 1564, 841,  1152, 2475, 3975, 2841, 1518,
        963, 2052, 3271, 2320, 1231, 585,  1239, 1964, 1385, 731}));
}

TEST(cuDNN_ConvTransposed, tune) {
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
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
    // check record
    auto kernelAttrs =
        KernelAttrs{Device::CUDA, conv->getOpType(), DataType::Float32};
    auto perfKey = PerfEngine::Key{kernelAttrs, conv->getOpPerfKey()};
    std::optional<PerfRecord> perfData =
        PerfEngine::getInstance().getPerfData(perfKey);
    ASSERT_TRUE(perfData.has_value());
}

} // namespace infini
