#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

template <class T>
void testElementWiseFp32(
    const std::function<void(void *, size_t, DataType)> &generator,
    // const Shape &shape0, const Shape &shape) {
    const Shape &shape0) {
    // // Runtime
    // Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    // auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    // // Build input data on CPU
    // Tensor inputCpu1 =
    //     make_ref<TensorObj>(shape0, DataType::Float16, cpuRuntime);
    // Tensor inputCpu2 =
    //     make_ref<TensorObj>(shape, DataType::Float16, cpuRuntime);
    // inputCpu1->dataMalloc();
    // inputCpu2->dataMalloc();
    // inputCpu1->setData(generator);
    // inputCpu2->setData(generator);

    // inputCpu1->print();
    // inputCpu1->printData();
    // inputCpu2->print();
    // inputCpu2->printData();
    // // NPU
    // Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    // auto inputNpu1 = npuGraph->cloneTensor(inputCpu1);
    // auto inputNpu2 = npuGraph->cloneTensor(inputCpu2);
    // auto npuOp = npuGraph->addOp<T>(inputNpu1, inputNpu2, nullptr);
    // npuGraph->dataMalloc();
    // inputNpu1->setData(generator);
    // inputNpu2->setData(generator);
    // npuRuntime->run(npuGraph);
    // auto outputNpu = npuOp->getOutput();
    // auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    // // Check
    // auto ExpectData = vector<float>{4., 4.}
    // // outputNpu2Cpu->print();
    // // outputNpu2Cpu->printData();
    // auto oCpu = gCpu->cloneTensor(npuOp->getOutput()); // move Data from gpu to cpu
    // oCpu->printData();                              //->printData
    // EXPECT_TRUE(oCpu->equalData(ExpectData));
    std::cout << "Start testElementWise" << std::endl;
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto inputShape = Shape{2, 2};
    auto ExpectData = vector<float>{4., 4., 4., 4.};
    auto inputCpu1 = gCpu->addTensor(inputShape, DataType::Float32);
    
    auto inputCpu2 = gCpu->addTensor(inputShape, DataType::Float32);
    gCpu->dataMalloc();

    inputCpu1->setData(generator);
    inputCpu2->setData(generator);
    inputCpu1->printData();
    std::cout << "CPU Inputs set" << std::endl;
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);

    auto inputNpu1 = npuGraph->cloneTensor(inputCpu1);
    auto inputNpu2 = npuGraph->cloneTensor(inputCpu2);

    std::cout << "NPU Inputs cloned" << std::endl;

    auto op = npuGraph->addOp<T>(inputNpu1, inputNpu2, nullptr);
    npuGraph->dataMalloc();

    inputNpu1->setData(generator);
    inputNpu2->setData(generator);

    std::cout << "NPU Inputs set" << std::endl;
    
    npuRuntime->run(npuGraph);

    std::cout << "NPU run completed" << std::endl;

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
    std::cout << "Test passed" << std::endl;
}

template <class T>
void testElementWiseFp16(
    const std::function<void(void *, size_t, DataType)> &generator,
    // const Shape &shape0, const Shape &shape) {
    const Shape &shape0) {
    
    std::cout << "Start testElementWise" << std::endl;
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto inputShape = Shape{2, 2};
    auto ExpectData = vector<float>{4., 4., 4., 4.};
    auto inputCpu1 = gCpu->addTensor(inputShape, DataType::Float16);
    
    auto inputCpu2 = gCpu->addTensor(inputShape, DataType::Float16);
    gCpu->dataMalloc();

    inputCpu1->setData(generator);
    inputCpu2->setData(generator);
    inputCpu1->printData();
    std::cout << "CPU Inputs set" << std::endl;
    auto npuRuntime = make_ref<ASCENDRuntimeObj>();
    Graph npuGraph = make_ref<GraphObj>(npuRuntime);

    auto inputNpu1 = npuGraph->cloneTensor(inputCpu1);
    auto inputNpu2 = npuGraph->cloneTensor(inputCpu2);

    std::cout << "NPU Inputs cloned" << std::endl;

    auto op = npuGraph->addOp<T>(inputNpu1, inputNpu2, nullptr);
    npuGraph->dataMalloc();

    inputNpu1->setData(generator);
    inputNpu2->setData(generator);

    std::cout << "NPU Inputs set" << std::endl;
    
    npuRuntime->run(npuGraph);

    std::cout << "NPU run completed" << std::endl;

    auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
    oCpu->printData();                              //->printData
    EXPECT_TRUE(oCpu->equalData(ExpectData));
    std::cout << "Test passed" << std::endl;
}

TEST(ascend_ElementWise, run) {
    aclInit(nullptr);
    // testElementWise<PowObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testElementWise<AddObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testElementWise<SubObj>(IncrementalGenerator(), Shape{1, 1, 48, 48},
    // Shape{1, 1, 1, 1});
    // testElementWise<MaximumObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
    //                             Shape{1, 2, 2, 3});
    // testElementWise<DivObj>(IncrementalGenerator(),
    // Shape{1}, Shape{1, 2, 2, 3});
    // testElementWise<MulObj>(IncrementalGenerator(),
    // Shape{1, 2, 2, 3});
    //testElementWiseFp32<AddObj>(ValGenerator<2>(), Shape{2,2});
    testElementWiseFp16<AddObj>(ValGenerator<2>(), Shape{2,2});
    aclFinalize();
}

} // namespace infini
