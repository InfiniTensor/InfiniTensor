#include "bang/bang_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/concat.h"

#include "test.h"

namespace infini {

template <class T>
void testConcat(const std::function<void(void *, size_t, DataType)> &generator,
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
    auto gpuOp =
        bangGraph->addOp<T>(TensorVec{inputGpu1, inputGpu2}, nullptr, 2);
    bangGraph->dataMalloc();
    inputGpu1->setData(generator);
    inputGpu2->setData(generator);
    bangRuntime->run(bangGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // Check
    inputCpu1->print();
    inputCpu1->printData();
    inputCpu2->print();
    inputCpu2->printData();
    outputGpu2Cpu->print();
    outputGpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(cnnl_Concat, run) {
    testConcat<ConcatObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
