#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
TEST(ElementWise, ShapeInference) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 4, 1, 6}, DataType::UInt32);
        Tensor i1 = g->addTensor({3, 1, 5, 1}, DataType::UInt32);
        auto op = g->addOp<AddObj>(i0, i1, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 4, 5, 6}));
    }
}

template <class T>
void testElementWiseCudnn(
    const std::function<void(void *, size_t, DataType)> &generator) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor acpu = make_ref<TensorObj>(Shape{1, 1, 1, 4}, DataType::Float32, cpuRuntime);
    acpu->dataMalloc();
    acpu->setData(generator);

    Tensor bcpu = make_ref<TensorObj>(Shape{1, 1, 3, 1}, DataType::Float32, cpuRuntime);
    bcpu->dataMalloc();
    bcpu->setData(generator);

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto a = g->cloneTensor(acpu);
    auto b = g->cloneTensor(bcpu);
    auto op = g->addOp<T>(a, b, nullptr);
    auto bop = g->addOp<DivObj>(a, b, nullptr);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto c = op->getOutput();
    auto bcast = bop->getOutput();
    auto ccpu = c->clone(cpuRuntime);
    acpu->printData();
    bcpu->printData();
    ccpu->printData();
    auto bcastcpu = bcast->clone(cpuRuntime);
    bcastcpu->printData();
    // cudaPrintTensor(c);
    //  check results on CPU
    EXPECT_TRUE(ccpu->equalData(bcastcpu));
}

TEST(ElementWise, CuDNN) {
    testElementWiseCudnn<AddObj>(IncrementalGenerator());
}

} // namespace infini
