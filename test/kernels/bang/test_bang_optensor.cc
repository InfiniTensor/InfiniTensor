#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testOptensor(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto bangRuntime = make_ref<BangRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu1->dataMalloc();
    inputCpu1->setData(generator);
    Tensor inputCpu2 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu2->dataMalloc();
    inputCpu2->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu1 = bangGraph->cloneTensor(inputCpu1);
    auto inputGpu2 = bangGraph->cloneTensor(inputCpu2);
    auto gpuOp = bangGraph->addOp<T>(inputGpu1, inputGpu2, nullptr);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu1, inputCpu2, nullptr);
    cpuGraph->dataMalloc();
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    outputCpu->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

TEST(cuDNN_OpTensor, run) {
    testOptensor<AddObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testOptensor<SubObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testOptensor<MulObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testOptensor<DivObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
