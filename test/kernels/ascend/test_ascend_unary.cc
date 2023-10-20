#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "ascend/ascend_runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testUnary(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu = xpuGraph->cloneTensor(inputCpu);
    auto gpuOp = xpuGraph->addOp<T>(inputGpu, nullptr);
    xpuGraph->dataMalloc();
    inputGpu->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu, 1e-6));
}


TEST(ascend_Unary, run) {
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
