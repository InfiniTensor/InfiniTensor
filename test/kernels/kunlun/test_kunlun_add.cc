#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testAdd(const std::function<void(void *, size_t, DataType)> &generator,
             const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    Tensor inputCpu2 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputGpu1 = xpuGraph->cloneTensor(inputCpu1);
    auto inputGpu2 = xpuGraph->cloneTensor(inputCpu2);
    auto gpuOp = xpuGraph->addOp<T>(inputGpu1, inputGpu2, nullptr);
    xpuGraph->dataMalloc();
    inputGpu1->setData(generator);
    inputGpu2->setData(generator);
    xpuRuntime->run(xpuGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu1, inputCpu2, nullptr);
    cpuGraph->addTensor(inputCpu1);
    cpuGraph->addTensor(inputCpu2);
    cpuGraph->dataMalloc();
    inputCpu1->setData(generator);
    inputCpu2->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(xpu_add, run) {
    testAdd<AddObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<SubObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<MulObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<DivObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<EqualObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<GreaterEqualObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<GreaterThanObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<LessEqualObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
    testAdd<LessThanObj>(IncrementalGenerator(), Shape{1, 1, 1, 30});
}

} // namespace infini
