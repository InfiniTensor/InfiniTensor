#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
template <class T>
void testResnet(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor acpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    acpu->dataMalloc();
    acpu->setData(generator);

    Tensor bcpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    bcpu->dataMalloc();
    bcpu->setData(generator);

    Tensor ccpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    ccpu->dataMalloc();
    ccpu->setData(generator);

    Graph g = make_ref<GraphObj>(cudaRuntime);
    Graph cg = make_ref<GraphObj>(cpuRuntime);

    auto a = g->cloneTensor(acpu);
    auto b = g->cloneTensor(bcpu);
    auto c = g->cloneTensor(ccpu);

    auto op = g->addOpWithOutputs<T>(a, b, c);
    op = g->addOpWithOutputs<T>(c, b, c);
    auto cop = cg->addOpWithOutputs<T>(acpu, bcpu, ccpu);
    cop = cg->addOpWithOutputs<T>(ccpu, bcpu, ccpu);

    // allocate CUDA memory
    g->dataMalloc();
    cg->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);
    cpuRuntime->run(cg);

    // clone CUDA output to CPU
    auto gpu2cpu = c->clone(cpuRuntime);
    // cudaPrintTensor(c);
    //  check results on CPU
    ccpu->printData();
    EXPECT_TRUE(gpu2cpu->equalData(ccpu));
}

TEST(cuDNN_ElementWise, run) {
    testResnet<AddObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3});
    testResnet<SubObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3});
    testResnet<MulObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testResnet<PowObj>(
    //     IncrementalGenerator(), Shape{1, 2, 2, 3});
    // testResnet<DivObj>(
    //     IncrementalGenerator(), Shape{1, 2, 2, 3});
}

} // namespace infini
