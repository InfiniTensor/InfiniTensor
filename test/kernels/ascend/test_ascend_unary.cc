#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testUnary(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu = npuGraph->cloneTensor(inputCpu);
    auto npuOp = npuGraph->addOp<T>(inputNpu, nullptr);
    npuGraph->dataMalloc();
    inputNpu->setData(generator);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputNpu2Cpu, 1e-3));
}

TEST(ascend_Unary, run) {
    aclInit(nullptr);
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<AbsObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<HardSwishObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SinObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<GeluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<CosObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<ACosObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<ATanObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<CeilObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<FloorObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<ExpObj>(IncrementalGenerators(), Shape{1, 2, 2, 3});
    testUnary<NegObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<ReciprocalObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SqrtObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<RoundObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    aclFinalize();
}

} // namespace infini
