#include "xpu/xpu_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testAdd(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu1->dataMalloc();
    inputCpu1->setData(generator);
    Tensor inputCpu2 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu2->dataMalloc();
    inputCpu2->setData(generator);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu1 = xpuGraph->cloneTensor(inputCpu1);
    auto inputGpu2 = xpuGraph->cloneTensor(inputCpu2);
    auto gpuOp = xpuGraph->addOp<T>(inputGpu1, inputGpu2, nullptr);
    xpuGraph->dataMalloc();
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu1, inputCpu2, nullptr);
    cpuGraph->dataMalloc();
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    //outputCpu->printData();
    //outputGpu2Cpu->printData();
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(xpu_add, run) {
    testAdd<AddObj>(IncrementalGenerator(), Shape{10, 256, 256, 3});
    testAdd<SubObj>(IncrementalGenerator(), Shape{10, 256, 256, 3});
    testAdd<MulObj>(IncrementalGenerator(), Shape{10, 256, 256, 3});
    testAdd<DivObj>(IncrementalGenerator(), Shape{10, 256, 256, 3});
}

} // namespace infini
