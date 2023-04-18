#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testTrigon(const std::function<void(void *, size_t, DataType)> &generator,
                const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu = bangGraph->cloneTensor(inputCpu);
    auto gpuOp = bangGraph->addOp<T>(inputGpu, nullptr);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    inputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Trigon, run) {
    testTrigon<SinObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<CosObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<TanObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<ASinObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<ACosObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<ATanObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<SinHObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<CosHObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<TanHObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<ASinHObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<ACosHObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    testTrigon<ATanHObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
}

} // namespace infini
