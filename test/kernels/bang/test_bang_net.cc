#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

void testNet(const std::function<void(void *, size_t, DataType)> &generator,
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
    auto gpuOp = bangGraph->addOp<AddObj>(inputGpu1, inputGpu2, nullptr);
    auto outputGpu = gpuOp->getOutput();
    auto gpuOp2 = bangGraph->addOp<SigmoidObj>(outputGpu, nullptr);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu2 = gpuOp2->getOutput();
    auto outputGpu2Cpu2 = outputGpu2->clone(cpuRuntime);
    // Check
    inputCpu2->printData();
    outputGpu2Cpu2->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Net, run) { testNet(IncrementalGenerator(), Shape{1, 2, 2, 3}); }

} // namespace infini
