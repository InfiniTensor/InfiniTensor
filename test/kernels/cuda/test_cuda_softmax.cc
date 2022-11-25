#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/softmax.h"
#include "test.h"
#include <cmath>
namespace infini {

TEST(cuDNN_Softmax, run_axis1) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 4}, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->copyin(vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<SoftmaxObj>(inputGpu, nullptr, 1);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    cudaPrintTensor(outputGpu);
    // Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(
        vector<float>{0.032058604, 0.08714432, 0.23688284, 0.6439143,
                      0.032058604, 0.08714432, 0.23688284, 0.6439143}));
}

TEST(cuDNN_Softmax, run_axis0) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 4}, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->copyin(vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<SoftmaxObj>(inputGpu, nullptr, 0);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    cudaPrintTensor(outputGpu);
    // Check
    EXPECT_TRUE(
        outputGpu2Cpu->equalData(vector<float>{0., 0., 0., 0., 1, 1, 1, 1}));
}

TEST(cuDNN_Softmax2, run_axis1) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 2, 2, 2}, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(IncrementalGenerator());

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<SoftmaxObj>(inputGpu, nullptr, 1);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    cudaPrintTensor(outputGpu);
    // Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(vector<float>{
        0.0179862, 0.0179862, 0.0179862, 0.0179862, 0.9820138, 0.9820138,
        0.9820138, 0.9820138, 0.0179862, 0.0179862, 0.0179862, 0.0179862,
        0.9820138, 0.9820138, 0.9820138, 0.9820138}));
}

TEST(cuDNN_Softmax2, run_axis2) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 2, 2, 2}, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(IncrementalGenerator());

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<SoftmaxObj>(inputGpu, nullptr, 2);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    cudaPrintTensor(outputGpu);
    // Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(vector<float>{
        0.1192029, 0.1192029, 0.8807971, 0.8807971, 0.1192029, 0.1192029,
        0.8807971, 0.8807971, 0.1192029, 0.1192029, 0.8807971, 0.8807971,
        0.1192029, 0.1192029, 0.8807971, 0.8807971}));
}

TEST(cuDNN_Softmax2, run_axis3) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu =
        make_ref<TensorObj>(Shape{2, 2, 2, 2}, DataType::Float32, cpuRuntime);
    inputCpu->dataMalloc();
    inputCpu->setData(IncrementalGenerator());

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<SoftmaxObj>(inputGpu, nullptr, 3);
    cudaGraph->dataMalloc();
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    cudaPrintTensor(outputGpu);
    // Check
    EXPECT_TRUE(outputGpu2Cpu->equalData(vector<float>{
        0.2689414, 0.7310586, 0.2689414, 0.7310586, 0.2689414, 0.7310586,
        0.2689414, 0.7310586, 0.2689414, 0.7310586, 0.2689414, 0.7310586,
        0.2689414, 0.7310586, 0.2689414, 0.7310586}));
}
} // namespace infini
