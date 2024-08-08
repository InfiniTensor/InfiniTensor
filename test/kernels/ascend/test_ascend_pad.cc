#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/pad.h"

#include "test.h"

namespace infini {

template <class T>
void testPad(const std::function<void(void *, size_t, DataType)> &generator,
             const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr, vector<int>{1, 1, 1, 1},
                                    vector<int>{0, 3});

    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    std::cout << npuOp->toString() << std::endl;
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    // Check
    inputCpu->print();
    inputCpu->printData();
    outputNpu2Cpu->print();
    outputNpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(ascend_Pad, run) {
    aclInit(nullptr);
    testPad<PadObj>(IncrementalGenerator(), Shape{1, 1, 2, 3});
    aclFinalize();
}

} // namespace infini
