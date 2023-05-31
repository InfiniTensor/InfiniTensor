#include "xpu/xpu_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testUnary(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    auto gpuOp = xpuGraph->addOp<T>(inputGpu, nullptr);
    xpuGraph->dataMalloc();
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->dataMalloc();
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(xdnn_Unary, run) {
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
