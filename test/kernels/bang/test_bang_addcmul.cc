#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testAddcmul(const std::function<void(void *, size_t, DataType)> &generator,
                 const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
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
    Tensor inputCpu3 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu3->dataMalloc();
    inputCpu3->setData(generator);

    // GPU
    Graph bangGraph = make_ref<GraphObj>(bangRuntime);
    auto inputGpu1 = bangGraph->cloneTensor(inputCpu1);
    auto inputGpu2 = bangGraph->cloneTensor(inputCpu2);
    auto inputGpu3 = bangGraph->cloneTensor(inputCpu3);
    float alpha = 1.1;
    auto gpuOp =
        bangGraph->addOp<T>(alpha, inputGpu1, inputGpu2, inputGpu3, nullptr);
    bangGraph->dataMalloc();
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // Check
    inputCpu1->printData();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_addcmul, run) {
    testAddcmul<AddcmulObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
