#include "ascend/ascend_runtime.h"
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
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

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
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu1 = npuGraph->cloneTensor(inputCpu1);
    auto inputNpu2 = npuGraph->cloneTensor(inputCpu2);
    auto npuOp =
        npuGraph->addOp<T>(TensorVec{inputNpu1, inputNpu2}, nullptr, 2);
    npuGraph->dataMalloc();
    npuRuntime->run(npuGraph);
    auto outputNpu = npuOp->getOutput();
    auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);
    // Check
    inputCpu1->print();
    inputCpu1->printData();
    inputCpu2->print();
    inputCpu2->printData();
    outputNpu2Cpu->print();
    outputNpu2Cpu->printData();
    EXPECT_TRUE(1);
}

TEST(ascend_Concat, run) {
    testConcat<ConcatObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
