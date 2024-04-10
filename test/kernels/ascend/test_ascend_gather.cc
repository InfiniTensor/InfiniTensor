#include "ascend/ascend_runtime.h"
#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/gather.h"

#include "test.h"

namespace infini {

TEST(ascend_Gather, run) {
    aclInit(nullptr);
    {
        // Runtime
        Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
        auto npuRuntime = make_ref<ASCENDRuntimeObj>();

        // Build input data on CPU
        Tensor inputCpu =
            make_ref<TensorObj>(Shape{3, 2}, DataType::Float32, cpuRuntime);
        Tensor indexCpu =
            make_ref<TensorObj>(Shape{2, 2}, DataType::Int32, cpuRuntime);

        // NPU
        Graph npuGraph = make_ref<GraphObj>(npuRuntime);
        auto inputNpu = npuGraph->cloneTensor(inputCpu);
        auto indexNpu = npuGraph->cloneTensor(indexCpu);
        auto npuOp = npuGraph->addOp<GatherObj>(inputNpu, indexNpu, nullptr, 0);
        npuGraph->dataMalloc();
        inputNpu->copyin(vector<float>{1, 2, 3, 4, 5, 6});
        indexNpu->copyin(vector<int>{0, 1, 1, 2});
        npuRuntime->run(npuGraph);
        auto outputNpu = npuOp->getOutput();
        auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

        // Check
        EXPECT_TRUE(
            outputNpu2Cpu->equalData(vector<float>{1, 2, 3, 4, 3, 4, 5, 6}));
    }
    {
        // Runtime
        Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
        auto npuRuntime = make_ref<ASCENDRuntimeObj>();

        // Build input data on CPU
        Tensor inputCpu =
            make_ref<TensorObj>(Shape{3, 3}, DataType::Float32, cpuRuntime);
        Tensor indexCpu =
            make_ref<TensorObj>(Shape{1, 2}, DataType::Int32, cpuRuntime);

        // NPU
        Graph npuGraph = make_ref<GraphObj>(npuRuntime);
        auto inputNpu = npuGraph->cloneTensor(inputCpu);
        auto indexNpu = npuGraph->cloneTensor(indexCpu);
        auto npuOp = npuGraph->addOp<GatherObj>(inputNpu, indexNpu, nullptr, 1);
        npuGraph->dataMalloc();
        inputNpu->setData(IncrementalGenerator());
        indexNpu->copyin(vector<int>{0, 2});
        npuRuntime->run(npuGraph);
        auto outputNpu = npuOp->getOutput();
        auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

        // Check
        EXPECT_TRUE(outputNpu2Cpu->equalData(vector<float>{0, 2, 3, 5, 6, 8}));
    }
    {
        // Runtime
        Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
        auto npuRuntime = make_ref<ASCENDRuntimeObj>();

        // Build input data on CPU
        Tensor inputCpu =
            make_ref<TensorObj>(Shape{3, 2}, DataType::Float32, cpuRuntime);
        Tensor indexCpu =
            make_ref<TensorObj>(Shape{2, 2}, DataType::Int64, cpuRuntime);

        // NPU
        Graph npuGraph = make_ref<GraphObj>(npuRuntime);
        auto inputNpu = npuGraph->cloneTensor(inputCpu);
        auto indexNpu = npuGraph->cloneTensor(indexCpu);
        auto npuOp = npuGraph->addOp<GatherObj>(inputNpu, indexNpu, nullptr, 0);
        npuGraph->dataMalloc();
        inputNpu->copyin(std::vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
        indexNpu->copyin(vector<int64_t>{0, 1, 1, 2});
        npuRuntime->run(npuGraph);
        auto outputNpu = npuOp->getOutput();
        auto outputNpu2Cpu = outputNpu->clone(cpuRuntime);

        // Check
        EXPECT_TRUE(outputNpu2Cpu->equalData(
            vector<float>{1.0, 1.2, 2.3, 3.4, 2.3, 3.4, 4.5, 5.7}));
    }
    aclFinalize();
}

} // namespace infini
