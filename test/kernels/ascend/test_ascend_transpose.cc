#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

template <class T>
void testTranspose(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const Shape &permute) {
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
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr, permute);
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
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

TEST(ascend_Unary, run) {
    aclInit(nullptr);
    testTranspose<TransposeObj>(IncrementalGenerator(), Shape{1, 1, 2, 3},
                                vector<int>{0, 1, 3, 2});
    aclFinalize();
}

} // namespace infini
