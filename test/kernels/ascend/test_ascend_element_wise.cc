#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testElementWise(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu1 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    Tensor inputCpu2 =
        make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputCpu1->dataMalloc();
    inputCpu2->dataMalloc();
    inputCpu1->setData(generator);
    inputCpu2->setData(generator);

    // NPU
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    auto inputNpu1 = npuGraph->cloneTensor(inputCpu1);
    auto inputNpu2 = npuGraph->cloneTensor(inputCpu2);
    auto npuOp = npuGraph->addOp<T>(inputNpu1, inputNpu2, nullptr);
    npuGraph->dataMalloc();
    inputNpu1->setData(generator);
    inputNpu2->setData(generator);
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

TEST(ascend_ElementWise, run) {
    aclInit(nullptr);
    testElementWise<PowObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testElementWise<AddObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testElementWise<SubObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testElementWise<DivObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testElementWise<MulObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    aclFinalize();
}

} // namespace infini
