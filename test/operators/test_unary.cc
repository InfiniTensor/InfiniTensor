#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;

template <class T>
void testUnary(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    // Runtime
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor inputcpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    inputcpu->dataMalloc();
    inputcpu->setData(generator);

    // GPU
    Graph cuda_g = make_ref<GraphObj>(cudaRuntime);
    auto inputgpu = cuda_g->cloneTensor(inputcpu);
    auto gpu_op = cuda_g->addOp<T>(inputgpu, nullptr);
    cuda_g->dataMalloc();
    cudaRuntime->run(cuda_g);
    auto outputgpu = gpu_op->getOutput();
    auto outputgpu2cpu = outputgpu->clone(cpuRuntime);
    // CPU
    Graph cpu_g = make_ref<GraphObj>(cpuRuntime);
    auto cpu_op = cpu_g->addOp<T>(inputcpu, nullptr);
    cpu_g->dataMalloc();
    cpuRuntime->run(cpu_g);
    auto outputcpu = cpu_op->getOutput();
    // Check
    EXPECT_TRUE(outputcpu->equalData(outputgpu2cpu));
}

TEST(Unary, CuDNN) {
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SoftmaxObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<AbsObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
