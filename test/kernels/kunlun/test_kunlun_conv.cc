#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

template <class T>
void testConv(const std::function<void(void *, size_t, DataType)> &generatorA,
              const std::function<void(void *, size_t, DataType)> &generatorB,
              const Shape &shapeA, const Shape &shapeB) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto xpuRuntime = make_ref<KUNLUNRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shapeA, DataType::Float32, cpuRuntime);
    Tensor inputCpu2 =
        make_ref<TensorObj>(shapeB, DataType::Float32, cpuRuntime);
    // MLU
    Graph xpuGraph = make_ref<GraphObj>(xpuRuntime);
    auto inputMlu1 = xpuGraph->cloneTensor(inputCpu1);
    auto inputMlu2 = xpuGraph->cloneTensor(inputCpu2);
    auto mluOp =
        xpuGraph->addOp<T>(inputMlu1, inputMlu2, nullptr, 1, 1, 1, 1, 1, 1);
    xpuGraph->dataMalloc();
    inputMlu1->setData(generatorA);
    inputMlu2->setData(generatorB);
    xpuRuntime->run(xpuGraph);
    auto outputXpu = mluOp->getOutput();
    auto outputXpu2Cpu = outputXpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    cpuGraph->addTensor(inputCpu1);
    cpuGraph->addTensor(inputCpu2);
    auto cpuOp =
        cpuGraph->addOp<T>(inputCpu1, inputCpu2, nullptr, 1, 1, 1, 1, 1, 1);
    cpuGraph->dataMalloc();
    inputCpu1->setData(generatorA);
    inputCpu2->setData(generatorB);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputXpu2Cpu));
}

TEST(xpu_Conv, run) {
    testConv<ConvObj>(IncrementalGenerator(), IncrementalGenerator(),
                      Shape{1, 3, 32, 32}, Shape{2, 3, 3, 3});
}

} // namespace infini
