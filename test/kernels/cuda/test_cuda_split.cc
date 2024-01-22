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

    auto inputGpu = gCuda->cloneTensor(input);
    auto op = gCuda->addOp<SplitObj>(inputGpu, std::nullopt, 1, 3);
    gCuda->dataMalloc();
    inputGpu->setData(IncrementalGenerator());

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

TEST(Split, CudaHigh) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({2, 6, 2, 1, 2}, DataType::Float32);
    gCpu->dataMalloc();
    input->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);
    auto op = gCuda->addOp<SplitObj>(inputGpu, std::nullopt, 1, 3);
    gCuda->dataMalloc();
    inputGpu->setData(IncrementalGenerator());

    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    EXPECT_EQ(op->getOutputs().size(), (size_t)3);
    auto o0Cpu = gCpu->cloneTensor(op->getOutput(0));
    auto o1Cpu = gCpu->cloneTensor(op->getOutput(1));
    auto o2Cpu = gCpu->cloneTensor(op->getOutput(2));
    EXPECT_TRUE(
        o0Cpu->equalData(vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 24., 25.,
                                       26., 27., 28., 29., 30., 31.}));
    EXPECT_TRUE(o1Cpu->equalData(vector<float>{8., 9., 10., 11., 12., 13., 14.,
                                               15., 32., 33., 34., 35., 36.,
                                               37., 38., 39.}));
    EXPECT_TRUE(o2Cpu->equalData(vector<float>{16., 17., 18., 19., 20., 21.,
                                               22., 23., 40., 41., 42., 43.,
                                               44., 45., 46., 47.}));
}

TEST(Split, SplitWithRatio) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({2, 6, 2, 1, 2}, DataType::Float32);
    gCpu->dataMalloc();
    input->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);
    vector<int> split = {2, 4};
    auto op = gCuda->addOp<SplitObj>(inputGpu, std::nullopt, 1, split);
    gCuda->dataMalloc();
    inputGpu->setData(IncrementalGenerator());

    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    EXPECT_EQ(op->getOutputs().size(), (size_t)2);
    auto o0Cpu = gCpu->cloneTensor(op->getOutput(0));
    auto o1Cpu = gCpu->cloneTensor(op->getOutput(1));
    EXPECT_TRUE(
        o0Cpu->equalData(vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 24., 25.,
                                       26., 27., 28., 29., 30., 31.}));
    EXPECT_TRUE(o1Cpu->equalData(
        vector<float>{8.,  9.,  10., 11., 12., 13., 14., 15., 16., 17., 18.,
                      19., 20., 21., 22., 23., 32., 33., 34., 35., 36., 37.,
                      38., 39., 40., 41., 42., 43., 44., 45., 46., 47.}));
}

TEST(Split, Cuda_dim0) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({2, 3}, DataType::Float32);
    gCpu->dataMalloc();
    input->setData(IncrementalGenerator());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);
    auto op = gCuda->addOp<SplitObj>(inputGpu, std::nullopt, 0, 2);
    gCuda->dataMalloc();
    inputGpu->setData(IncrementalGenerator());

    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    EXPECT_EQ(op->getOutputs().size(), (size_t)2);
    auto o0Cpu = gCpu->cloneTensor(op->getOutput(0));
    auto o1Cpu = gCpu->cloneTensor(op->getOutput(1));
    EXPECT_TRUE(o0Cpu->equalData(vector<float>{0, 1, 2}));
    EXPECT_TRUE(o1Cpu->equalData(vector<float>{3, 4, 5}));
}
//----------------
TEST(SplitFp16, CudaHigh) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph gCpu = make_ref<GraphObj>(runtime);

    auto input = gCpu->addTensor({2, 6, 2, 1, 2}, DataType::Float16);
    gCpu->dataMalloc();
    input->setData(ValGenerator<2>());

    auto cudaRuntime = make_ref<CudaRuntimeObj>();
    Graph gCuda = make_ref<GraphObj>(cudaRuntime);

    auto inputGpu = gCuda->cloneTensor(input);
    auto op = gCuda->addOp<SplitObj>(inputGpu, std::nullopt, 1, 3);
    gCuda->dataMalloc();
    inputGpu->setData(ValGenerator<2>());

    cudaRuntime->run(gCuda);

    //  copy output from CUDA to CPU
    EXPECT_EQ(op->getOutputs().size(), (size_t)3);
    auto o0Cpu = gCpu->cloneTensor(op->getOutput(0));
    auto o1Cpu = gCpu->cloneTensor(op->getOutput(1));
    auto o2Cpu = gCpu->cloneTensor(op->getOutput(2));
    EXPECT_TRUE(o0Cpu->equalData(vector<float>{
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.}));
    EXPECT_TRUE(o1Cpu->equalData(vector<float>{
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.}));
    EXPECT_TRUE(o2Cpu->equalData(vector<float>{
        2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.}));
}
} // namespace infini
