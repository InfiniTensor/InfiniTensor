#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/split.h"

#include "test.h"

namespace infini {

TEST(Split, Cuda) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({2, 10, 2, 1}, DataType::Float32);
    gCpu->dataMalloc();
    input->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto op =
        gCuda->addOp<SplitObj>(gCuda->cloneTensor(input), std::nullopt, 1, 3);
    gCuda->dataMalloc();
    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    EXPECT_EQ(op->getOutputs().size(), (size_t)3);
    auto o0Cpu = gCpu->cloneTensor(op->getOutput(0));
    auto o1Cpu = gCpu->cloneTensor(op->getOutput(1));
    auto o2Cpu = gCpu->cloneTensor(op->getOutput(2));
    EXPECT_TRUE(o0Cpu->equalData(
        vector<float>{0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25}));
    EXPECT_TRUE(o1Cpu->equalData(
        vector<float>{6, 7, 8, 9, 10, 11, 26, 27, 28, 29, 30, 31}));
    EXPECT_TRUE(o2Cpu->equalData(vector<float>{
        12, 13, 14, 15, 16, 17, 18, 19, 32, 33, 34, 35, 36, 37, 38, 39}));
}

} // namespace infini
