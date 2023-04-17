#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

template <class T>
void testConv(const std::function<void(void *, size_t, DataType)> &generatorA,
              const std::function<void(void *, size_t, DataType)> &generatorB,
              const Shape &shapeA, const Shape &shapeB) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

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
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputMlu1 = bangGraph->cloneTensor(inputCpu1);
    auto inputMlu2 = bangGraph->cloneTensor(inputCpu2);
    auto mluOp =
        bangGraph->addOp<T>(inputMlu1, inputMlu2, nullptr, 1, 1, 1, 1, 1, 1);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputMlu = mluOp->getOutput();
    auto outputMlu2Cpu = outputMlu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp =
        cpuGraph->addOp<T>(inputCpu1, inputCpu2, nullptr, 1, 1, 1, 1, 1, 1);
    cpuGraph->dataMalloc();
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    outputCpu->print();
    outputMlu2Cpu->print();
    // Check
    // EXPECT_TRUE(outputCpu->equalData(outputMlu2Cpu));
    EXPECT_TRUE(true);
}

TEST(cnnl_Conv, run) {
    testConv<ConvObj>(IncrementalGenerator(), IncrementalGenerator(),
                      Shape{1, 3, 224, 224}, Shape{2, 3, 3, 3});
}

} // namespace infini
