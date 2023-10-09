#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/pooling.h"
#include "xpu/xpu_runtime.h"

#include "test.h"

namespace infini {

template <class T>
void testPooling(const std::function<void(void *, size_t, DataType)> &generator,
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
    auto gpuOp = xpuGraph->addOp<T>(inputGpu, nullptr, 3, 3, 1, 1, 0, 0, 2, 2);
    xpuGraph->dataMalloc();
    inputGpu->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);

    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    cpuGraph->addTensor(inputCpu);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr, 3, 3, 1, 1, 0, 0, 2, 2);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(xdnn_Pooling, run) {
    testPooling<MaxPoolObj>(IncrementalGenerator(), Shape{1, 1, 5, 5});
    testPooling<AvgPoolObj>(IncrementalGenerator(), Shape{1, 1, 5, 5});
}

} // namespace infini
