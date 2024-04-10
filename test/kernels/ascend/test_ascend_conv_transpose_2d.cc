#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

void testConvTransposedAclnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    std::vector<float> ansVec) {
    const auto &[N, C, H, W, F, R, S] = tuple{1, 1, 2, 2, 1, 4, 4};
    const int stride = 1, padding = 0, dilation = 1;
    // Construct Runtime and graph for CPU and CUDA
    Runtime cpu = NativeCpuRuntimeObj::getInstance(); // CPUruntime is singleton
    Graph gCpu = make_ref<GraphObj>(cpu);
    Runtime npu = make_ref<ASCENDRuntimeObj>();
    Graph gNpu = make_ref<GraphObj>(npu);
    // Set input data on CPU in a CPU Graph
    Tensor i0Cpu = gCpu->addTensor({N, F, H, H}, DataType::Float32);
    Tensor w0Cpu = gCpu->addTensor({F, C, R, S}, DataType::Float32);
    // Malloc data for all tensors in a graph. Do we need implicit allocation?
    gCpu->dataMalloc();
    i0Cpu->setData(generator);
    w0Cpu->setData(generator);

    // Copy input tensors from CPU to CUDA
    Tensor i0Npu = gNpu->cloneTensor(i0Cpu);
    Tensor w0Npu = gNpu->cloneTensor(w0Cpu);
    // Build CUDA graph
    auto conv = gNpu->addOp<ConvTransposed2dObj>(i0Npu, w0Npu, nullptr, padding,
                                                 padding, stride, stride,
                                                 dilation, dilation);
    gNpu->dataMalloc();
    i0Npu->setData(generator);
    w0Npu->setData(generator);
    // Execute on CUDA
    npu->run(gNpu);
    // copy output from CUDA to CPU
    auto o0Cpu = gCpu->cloneTensor(conv->getOutput());
    // check results on CPU
    o0Cpu->printData();
    EXPECT_TRUE(o0Cpu->equalData(ansVec));
}

TEST(ascend_ConvTransposed, run) {
    testConvTransposedAclnn(
        IncrementalGenerator(),
        std::vector<float>{0.,  0.,  1.,  2.,  3.,  0.,  6.,  12., 18.,
                           16., 8.,  30., 36., 42., 32., 16., 54., 60.,
                           66., 48., 24., 62., 67., 72., 45.});
}

} // namespace infini
