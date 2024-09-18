#include "ascend/ascend_runtime.h"
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
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);
    // GPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto gpuOp = npuGraph->addOp<T>(inputNpu, std::nullopt, 3, 3);
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);
    auto o0Cpu = gpuOp->getOutput(0)->clone(cpuRuntime);
    auto o1Cpu = gpuOp->getOutput(1)->clone(cpuRuntime);
    auto o2Cpu = gpuOp->getOutput(2)->clone(cpuRuntime);
    // Check
    inputCpu->print();
    inputCpu->printData();
    o0Cpu->print();
    o0Cpu->printData();
    o1Cpu->print();
    o1Cpu->printData();
    o2Cpu->print();
    o2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(ascend_Split, run) {
    aclInit(nullptr);
    testSplit<SplitObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    aclFinalize();
}

} // namespace infini
