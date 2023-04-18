#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/split.h"

#include "test.h"

namespace infini {

template <class T>
void testSplit(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu1->dataMalloc();
    inputCpu1->setData(generator);
    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu1 = bangGraph->cloneTensor(inputCpu1);
    auto gpuOp = bangGraph->addOp<T>(inputGpu1, std::nullopt, 3, 3);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto o0Cpu = gpuOp->getOutput(0)->clone(cpuRuntime);
    auto o1Cpu = gpuOp->getOutput(1)->clone(cpuRuntime);
    auto o2Cpu = gpuOp->getOutput(2)->clone(cpuRuntime);
    // Check
    inputCpu1->print();
    inputCpu1->printData();
    o0Cpu->print();
    o0Cpu->printData();
    o1Cpu->print();
    o1Cpu->printData();
    o2Cpu->print();
    o2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Split, run) {
    testSplit<SplitObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
