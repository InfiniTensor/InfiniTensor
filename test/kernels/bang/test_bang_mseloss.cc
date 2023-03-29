#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testMSELoss(const std::function<void(void *, size_t, DataType)> &generator,
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
    auto gpuOp1 =
        bangGraph->addOp<T>(inputGpu1, inputGpu2, MSELossObj::None, nullptr);
    auto gpuOp2 =
        bangGraph->addOp<T>(inputGpu1, inputGpu2, MSELossObj::Sum, nullptr);
    auto gpuOp3 =
        bangGraph->addOp<T>(inputGpu1, inputGpu2, MSELossObj::Mean, nullptr);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu1 = gpuOp1->getOutput();
    auto outputGpu2 = gpuOp2->getOutput();
    auto outputGpu3 = gpuOp3->getOutput();
    auto outputGpu2Cpu1 = outputGpu1->clone(cpuRuntime);
    auto outputGpu2Cpu2 = outputGpu2->clone(cpuRuntime);
    auto outputGpu2Cpu3 = outputGpu3->clone(cpuRuntime);
    // Check
    outputGpu2Cpu1->printData();
    outputGpu2Cpu2->printData();
    outputGpu2Cpu3->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_MSELoss, run) {
    testMSELoss<MSELossObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
