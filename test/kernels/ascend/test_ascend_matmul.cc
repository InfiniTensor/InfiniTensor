#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/matmul.h"

#include "test.h"

namespace infini {

template <class T>
void testMatmul(const std::function<void(void *, size_t, DataType)> &generatorA,
                const std::function<void(void *, size_t, DataType)> &generatorB,
                bool transA, bool transB, const Shape &shapeA,
                const Shape &shapeB) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shapeA, DataType::Float32, cpuRuntime);
    Tensor inputCpu2 =
        make_ref<TensorObj>(shapeB, DataType::Float32, cpuRuntime);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu1 = npuGraph->cloneTensor(inputCpu1);
    auto inputNpu2 = npuGraph->cloneTensor(inputCpu2);
    auto npuOp = npuGraph->addOp<T>(inputNpu1, inputNpu2, nullptr);
    npuGraph->dataMalloc();
    inputNpu1->setData(generatorA);
    inputNpu2->setData(generatorB);
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu1, inputCpu2, nullptr);
    cpuGraph->addTensor(inputCpu1);
    cpuGraph->addTensor(inputCpu2);
    cpuGraph->dataMalloc();
    inputCpu1->setData(generatorA);
    inputCpu2->setData(generatorB);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();

    // Check
    EXPECT_TRUE(outputCpu->equalData(outputNpu2Cpu));
}

TEST(ascend_Matmul, run) {
    testMatmul<MatmulObj>(IncrementalGenerator(), IncrementalGenerator(), false,
                          false, Shape{1, 2, 3}, Shape{1, 3, 4});
}

} // namespace infini
