#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "operators/conv.h"
#include "kunlun/kunlun_kernel_without_config.h"
#include "kunlun/kunlun_runtime.h"

#include "test.h"

namespace infini {

void testConvTransposedXdnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    const auto &[N, C, H, W, F, R, S] = tuple{1, 1, 2, 2, 1, 4, 4};
    const int stride = 1, padding = 0, dilation = 1;
    // Construct Runtime and graph for CPU and XPU
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime xpu = make_ref<KUNLUNRuntimeObj>();
    Graph gXpu = make_ref<GraphObj>(xpu);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({N, F, H, H}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({F, C, R, S}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to XPU
    Tensor i0Xpu = gXpu->cloneTensor(i0Cpu);
    Tensor w0Xpu = gXpu->cloneTensor(w0Cpu);
    // Build XPU graph
    auto conv = gXpu->addOp<ConvTransposed2dObj>(i0Xpu, w0Xpu, nullptr, padding,
                                                 padding, stride, stride,
                                                 dilation, dilation);
    gXpu->dataMalloc();
    i0Xpu->setData(generator);
    w0Xpu->setData(generator);
    // Execute on XPU
    xpu->run(gXpu);
    // copy output from XPU to CPU
    auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
    // check results on CPU
    EXPECT_TRUE(o0Cpu->equalData(ansVec));
}

void testConvTransposedNHWCXdnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    const auto &[N, C, H, W, F, R, S] = tuple{1, 1, 2, 2, 1, 4, 4};
    const int stride = 1, padding = 0, dilation = 1;
    // Construct Runtime and graph for CPU and XPU
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime xpu = make_ref<KUNLUNRuntimeObj>();
    Graph gXpu = make_ref<GraphObj>(xpu);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({N, H, W, F}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({F, R, S, C}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to XPU
    Tensor i0Xpu = gXpu->cloneTensor(i0Cpu);
    Tensor w0Xpu = gXpu->cloneTensor(w0Cpu);
    // Build XPU graph
    auto conv = gXpu->addOp<ConvTransposed2dNHWCObj>(
        i0Xpu, w0Xpu, nullptr, padding, padding, stride, stride, dilation,
        dilation);
    gXpu->dataMalloc();
    i0Xpu->setData(generator);
    w0Xpu->setData(generator);
    // Execute on XPU
    xpu->run(gXpu);
    // copy output from XPU to CPU
    auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
    // check results on CPU
    EXPECT_TRUE(o0Cpu->equalData(ansVec));
}

TEST(XPU_ConvTransposed, run) {
    testConvTransposedXdnn(IncrementalGenerator(),
                           vector<float>{0.,  0.,  1.,  2.,  3.,  0.,  6.,
                                         12., 18., 16., 8.,  30., 36., 42.,
                                         32., 16., 54., 60., 66., 48., 24.,
                                         62., 67., 72., 45.});
}

TEST(XPU_ConvTransposedNHWC, run) {
    testConvTransposedNHWCXdnn(IncrementalGenerator(),
                               vector<float>{0.,  0.,  1.,  2.,  3.,  0.,  6.,
                                             12., 18., 16., 8.,  30., 36., 42.,
                                             32., 16., 54., 60., 66., 48., 24.,
                                             62., 67., 72., 45.});
}

TEST(XPU_ConvTransposed, run1) {
    // Construct Runtime and graph for CPU and XPU
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime xpu = make_ref<KUNLUNRuntimeObj>();
    Graph gXpu = make_ref<GraphObj>(xpu);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({1, 2, 3, 3}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({2, 2, 3, 3}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(IncrementalGenerator());
    w0Cpu->setData(IncrementalGenerator());

    // Copy input tensors from CPU to XPU
    Tensor i0Xpu = gXpu->cloneTensor(i0Cpu);
    Tensor w0Xpu = gXpu->cloneTensor(w0Cpu);
    // Build XPU graph
    auto conv = gXpu->addOp<ConvTransposed2dObj>(i0Xpu, w0Xpu, nullptr, 0, 0);
    gXpu->dataMalloc();
    i0Xpu->setData(IncrementalGenerator());
    w0Xpu->setData(IncrementalGenerator());
    // Execute on XPU
    xpu->run(gXpu);
    // copy output from XPU to CPU
    auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
    // check results on CPU
    EXPECT_TRUE(o0Cpu->equalData(vector<float>{
        162, 351,  569,  413,  224,  405,  876,  1417, 1024, 553,
        747, 1611, 2598, 1869, 1005, 639,  1368, 2191, 1564, 835,
        396, 843,  1343, 953,  506,  243,  531,  866,  629,  341,
        621, 1344, 2173, 1564, 841,  1152, 2475, 3975, 2841, 1518,
        963, 2052, 3271, 2320, 1231, 585,  1239, 1964, 1385, 731}));
}

} // namespace infini
