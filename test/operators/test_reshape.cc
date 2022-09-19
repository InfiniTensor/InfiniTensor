#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/reshape.h"

#include "test.h"

namespace infini {

TEST(Reshape, ShapeInference) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReshapeObj>(i, nullptr, Shape{3, 2, 4, 3});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 2, 4, 3}));
    }
}
TEST(Flatten, ShapeInference) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<FlattenObj>(i, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{72}));
    }
}

TEST(Identity, ShapeInference) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<IdentityObj>(i, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 3, 4}));
    }
}

TEST(Reshape, Cuda) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<ReshapeObj>(i, nullptr, Shape{3, 2, 4, 3});

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(icpu));
}

TEST(Flatten, Cuda) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<FlattenObj>(i, nullptr);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(icpu));
}

TEST(Identity, Cuda) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor icpu =
        make_ref<TensorObj>(Shape{2, 3, 3, 4}, DataType::Float32, cpuRuntime);
    icpu->dataMalloc();
    icpu->setData(IncrementalGenerator());

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i = g->cloneTensor(icpu);
    auto op = g->addOp<IdentityObj>(i, nullptr);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o = op->getOutput();
    auto ocpu = o->clone(cpuRuntime);

    //  check results on CPU
    EXPECT_TRUE(ocpu->equalData(icpu));
}
} // namespace infini