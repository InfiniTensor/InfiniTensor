#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

template <class T>
void testUnary(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);

    // GPU
    Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
    auto inputGpu = cudaGraph->cloneTensor(inputCpu);
    auto gpuOp = cudaGraph->addOp<T>(inputGpu, nullptr);
    cudaGraph->dataMalloc();
    inputGpu->setData(generator);
    cudaRuntime->run(cudaGraph);
    auto outputGpu = gpuOp->getOutput();
    auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);
    // CPU
    Graph cpuGraph = make_ref<GraphObj>(cpuRuntime);
    auto cpuOp = cpuGraph->addOp<T>(inputCpu, nullptr);
    cpuGraph->addTensor(inputCpu);
    cpuGraph->dataMalloc();
    inputCpu->setData(generator);
    cpuRuntime->run(cpuGraph);
    auto outputCpu = cpuOp->getOutput();
    // Check
    EXPECT_TRUE(outputCpu->equalData(outputGpu2Cpu));
}

// template <class T>
// void testCast(const std::function<void(void *, size_t, DataType)> &generator,
//               const Shape &shape, vector<float> ansVec) {
//     // Runtime
//     Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
//     auto cudaRuntime = make_ref<CudaRuntimeObj>();

//     // Build input data on CPU
//     Tensor inputCpu = make_ref<TensorObj>(shape, DataType::Float32,
//     cpuRuntime); inputCpu->dataMalloc(); inputCpu->setData(generator);

//     // GPU
//     Graph cudaGraph = make_ref<GraphObj>(cudaRuntime);
//     auto inputGpu = cudaGraph->cloneTensor(inputCpu);
//     auto gpuOp =
//         cudaGraph->addOp<T>(inputGpu, nullptr, CastType::Float2Float16);
//     cudaGraph->dataMalloc();
//     inputGpu->setData(generator);
//     cudaRuntime->run(cudaGraph);
//     auto outputGpu = gpuOp->getOutput();
//     auto outputGpu2Cpu = outputGpu->clone(cpuRuntime);

//     inputCpu->printData();
//     outputGpu2Cpu->printData();
//     EXPECT_TRUE(outputGpu2Cpu->equalData(ansVec));
// }

// TEST(LeakyRelu, Cuda_WithAlpha) {
//     Runtime runtime = NativeCpuRuntimeObj::getInstance();
//     Graph gCpu = make_ref<GraphObj>(runtime);

//     auto input = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
//     gCpu->dataMalloc();
//     input->copyin(vector<float>{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0, -1.5,
//                                 -1.0, 1.0, 2.0, 3.0});

//     auto cudaRuntime = make_ref<CudaRuntimeObj>();
//     Graph gCuda = make_ref<GraphObj>(cudaRuntime);

//     auto inputGpu = gCuda->cloneTensor(input);

//     float alpha = 0.01;
//     auto op = gCuda->addOp<LeakyReluObj>(inputGpu, nullptr, alpha);
//     gCuda->dataMalloc();
//     inputGpu->copyin(vector<float>{-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -2.0,
//     -1.5,
//                                    -1.0, 1.0, 2.0, 3.0});
//     cudaRuntime->run(gCuda);

//     auto oCpu = gCpu->cloneTensor(op->getOutput());
//     oCpu->printData();
//     EXPECT_TRUE(
//         oCpu->equalData(vector<float>{-0.01, -0.005, 0.0, 0.5, 1.0, 1.5,
//         -0.02,
//                                       -0.015, -0.01, 1.0, 2.0, 3.0}));
// }

// TEST(Elu, Cuda) {
//     Runtime runtime = NativeCpuRuntimeObj::getInstance();
//     Graph gCpu = make_ref<GraphObj>(runtime);

//     auto input = gCpu->addTensor({2, 2, 3, 1}, DataType::Float32);
//     gCpu->dataMalloc();
//     input->setData(IncrementalGenerator());

//     auto cudaRuntime = make_ref<CudaRuntimeObj>();
//     Graph gCuda = make_ref<GraphObj>(cudaRuntime);

//     auto inputGpu = gCuda->cloneTensor(input);
//     auto op = gCuda->addOp<EluObj>(inputGpu, nullptr, 1.0f);
//     gCuda->dataMalloc();
//     inputGpu->setData(IncrementalGenerator());
//     cudaRuntime->run(gCuda);

//     auto oCpu = gCpu->cloneTensor(op->getOutput());
//     oCpu->printData();
//     EXPECT_TRUE(oCpu->equalData(
//         vector<float>{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.}));
// }

TEST(cuDNN_Unary, run) {
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<SiluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<AbsObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<HardSigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<HardSwishObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<SqrtObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<NegObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testUnary<ErfObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testCast<CastObj>(IncrementalGenerator(), Shape{8, 1},
    //                   vector<float>{0, 1, 2, 3, 4, 5, 6, 7});
    // // more shapes
    // testUnary<SqrtObj>(IncrementalGenerator(), Shape{13});
    // testUnary<SqrtObj>(IncrementalGenerator(), Shape{4, 3});
    // testUnary<SqrtObj>(IncrementalGenerator(), Shape{2, 3, 4, 5, 6});

    // testUnary<GeluObj>(IncrementalGenerator(), Shape{1});
    // testUnary<GeluObj>(IncrementalGenerator(), Shape{1, 2});
    // testUnary<GeluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
