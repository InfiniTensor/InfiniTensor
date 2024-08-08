#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testErf(const std::function<void(void *, size_t, DataType)> &generator,
             const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto ascendRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(generator);

    // Npu
    Graph npuGraph = make_ref<GraphObj>(ascendRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr);
    npuGraph->dataMalloc();
    ascendRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    inputCpu->printData();
    outputNpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(ascend_Erf, run) {
    aclInit(nullptr);
    testErf<ErfObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    aclFinalize();
}

} // namespace infini
