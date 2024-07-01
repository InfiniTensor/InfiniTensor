#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/gather.h"  // 假设有对应的头文件

#include "test.h"

namespace infini {

void testGatherElementsFp16(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape0) {

    aclInit(nullptr);
    {
        std::cout << "Start testElementWise" << std::endl;
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph gCpu = make_ref<GraphObj>(runtime);

        Tensor testCpu = make_ref<TensorObj>(shape0, DataType::Float32, runtime);
        testCpu->dataMalloc();
        testCpu->setData(generator);
        testCpu->printData();
        auto inputShape = Shape{2, 2, 3};
        // auto ExpectData = generator();
        // ExpectData->printData();
        // auto ExpectData = vector<float>{2., 1., 0., 4., 3., 5., 8., 7., 6., 10., 9., 11.};
        auto ExpectData = vector<float>{2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.};
        auto inputCpu = gCpu->addTensor(inputShape, DataType::Float16);
        auto indexCpu = gCpu->addTensor(inputShape, DataType::Int64);
        gCpu->dataMalloc();

        inputCpu->setData(generator);
        indexCpu->copyin(vector<int64_t>{2, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2});
        inputCpu->printData();
        indexCpu->printData();
        std::cout << "CPU Inputs set" << std::endl;
        auto npuRuntime = make_ref<ASCENDRuntimeObj>();
        Graph npuGraph = make_ref<GraphObj>(npuRuntime);

        auto inputNpu = npuGraph->cloneTensor(inputCpu);
        auto indexNpu = npuGraph->cloneTensor(indexCpu);

        std::cout << "NPU Inputs cloned" << std::endl;

        auto op = npuGraph->addOp<GatherElementsObj>(inputNpu, indexNpu, nullptr, 2);
        npuGraph->dataMalloc();

        inputNpu->setData(generator);
        indexNpu->copyin(vector<int64_t>{2, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2});;

        std::cout << "NPU Inputs set" << std::endl;
        
        npuRuntime->run(npuGraph);

        std::cout << "NPU run completed" << std::endl;

        auto oCpu = gCpu->cloneTensor(op->getOutput()); // move Data from gpu to cpu
        oCpu->printData();                              //->printData
        EXPECT_TRUE(oCpu->equalData(ExpectData));
        std::cout << "Test2 passed" << std::endl;
    }
}

TEST(ascend_GatherElements, run) {
    aclInit(nullptr);

    // {
    //     // Runtime
    //     Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    //     auto npuRuntime = make_ref<ASCENDRuntimeObj>();

    //     // Build input data on CPU
    //     Tensor inputCpu =
    //         make_ref<TensorObj>(Shape{2, 3}, DataType::Float32, cpuRuntime);
    //     Tensor indexCpu =
    //         make_ref<TensorObj>(Shape{2, 3}, DataType::Int32, cpuRuntime);

    //     // NPU
    //     Graph npuGraph = make_ref<GraphObj>(npuRuntime);
    //     auto inputNpu = npuGraph->cloneTensor(inputCpu);
    //     auto indexNpu = npuGraph->cloneTensor(indexCpu);
    //     auto npuOp = npuGraph->addOp<GatherElementsObj>(inputNpu, indexNpu, nullptr, 1);
    //     npuGraph->dataMalloc();
    //     inputNpu->copyin(vector<float>{1, 2, 3, 4, 5, 6});
    //     indexNpu->copyin(vector<int32_t>{2, 1, 0, 0, 2, 1});
    //     npuRuntime->run(npuGraph);
    //     auto outputNpu = npuOp->getOutput();
    //     auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

    //     // Check
    //     EXPECT_TRUE(outputNpu2Cpu->equalData(vector<float>{3, 2, 1, 4, 6, 5}));
    // }

    {
        // Runtime
        Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
        auto npuRuntime = make_ref<ASCENDRuntimeObj>();

        // Build input data on CPU
        Tensor inputCpu =
            make_ref<TensorObj>(Shape{2, 2, 3}, DataType::Float32, cpuRuntime);
        Tensor indexCpu =
            make_ref<TensorObj>(Shape{2, 2, 3}, DataType::Int64, cpuRuntime);

        // NPU
        Graph npuGraph = make_ref<GraphObj>(npuRuntime);
        auto inputNpu = npuGraph->cloneTensor(inputCpu);
        auto indexNpu = npuGraph->cloneTensor(indexCpu);
        auto npuOp = npuGraph->addOp<GatherElementsObj>(inputNpu, indexNpu, nullptr, 2);
        npuGraph->dataMalloc();
        inputNpu->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        indexNpu->copyin(vector<int64_t>{2, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2});
        npuRuntime->run(npuGraph);
        auto outputNpu = npuOp->getOutput();
        auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

        // Check
        EXPECT_TRUE(outputNpu2Cpu->equalData(vector<float>{3, 2, 1, 5, 4, 6, 9, 8, 7, 11, 10, 12}));
        std::cout << "Test1 passed" << std::endl;
    }

    testGatherElementsFp16(IncrementalGenerator(), Shape{2,2,3});
    aclFinalize();
}

} // namespace infini
