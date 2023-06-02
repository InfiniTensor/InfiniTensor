#include "xpu/xpu_runtime.h"
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
    auto xpuRuntime = make_ref<XPURuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shapeA, DataType::Float32, cpuRuntime);
    inputCpu1->dataMalloc();
    inputCpu1->setData(generatorA);
    Tensor inputCpu2 =
        make_ref<TensorObj>(shapeB, DataType::Float32, cpuRuntime);
    inputCpu2->dataMalloc();
    inputCpu2->setData(generatorB);

    // MLU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputMlu1 = xpuGraph->cloneTensor(inputCpu1);
    auto inputMlu2 = xpuGraph->cloneTensor(inputCpu2);
    auto mluOp = xpuGraph->addOp<T>(inputMlu1, inputMlu2, nullptr);
    xpuGraph->dataMalloc();
    xpuRuntime->run(xpuGraph);
    auto outputMlu = mluOp->getOutput();
    auto outputMlu2Cpu = outputMlu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu1, inputCpu2, nullptr);
    cpuGraph->dataMalloc();
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    outputCpu->print();
    outputMlu2Cpu->print();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputMlu2Cpu));
}

TEST(xpu_Matmul, run) {
    testMatmul<MatmulObj>(IncrementalGenerator(), IncrementalGenerator(), false,
                          false, Shape{2, 3}, Shape{3, 4});
}

} // namespace infini