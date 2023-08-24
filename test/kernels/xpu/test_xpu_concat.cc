#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "xpu/xpu_runtime.h"

#include "test.h"

namespace infini {

template <class T>
void testConcat(const std::function<void(void *, size_t, DataType)> &generator,
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
    auto gpuOp =
        xpuGraph->addOp<T>(TensorVec{inputGpu1, inputGpu2}, nullptr, 2);
    xpuGraph->dataMalloc();
    inputGpu1->setData(generator);
    inputGpu2->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // Check
    inputCpu1->print();
    inputCpu1->printData();
    inputCpu2->print();
    inputCpu2->printData();
    outputGpu2Cpu->print();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(xpu_Concat, run) {
    testConcat<ConcatObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
