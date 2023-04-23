#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "operators/any.h"
#include "test.h"
using namespace infini;
using namespace std;

TEST(Any, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    vector<int> attr;
    string kernelName = "fake_kernel_name";
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 2, 3}, DataType::Float32);
        Tensor i1 = g->addTensor({2, 2, 3}, DataType::Float32);
        Tensor o0 = g->addTensor({3, 2, 3}, DataType::Float32);
        auto anyOp = g->addOpWithOutputs<AnyObj>(
            TensorVec{i0, i1}, TensorVec{o0}, kernelName, attr);
        EXPECT_TRUE(anyOp->getOutputs().size() == 1);
        EXPECT_EQ(anyOp->getOutput()->getDims(), (Shape{3, 2, 3}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({1, 2, 3}, DataType::Float32);
        Tensor i1 = g->addTensor({2, 2, 3}, DataType::Float32);
        Tensor o0 = g->addTensor({2, 2, 3}, DataType::Float32);
        Tensor o1 = g->addTensor({1, 2, 3}, DataType::Float32);
        auto anyOp = g->addOpWithOutputs<AnyObj>(
            TensorVec{i0, i1}, TensorVec{o0, o1}, kernelName, attr);
        EXPECT_TRUE(anyOp->getOutputs().size() == 2);
        EXPECT_EQ(anyOp->getOutput(0)->getDims(), (Shape{2, 2, 3}));
        EXPECT_EQ(anyOp->getOutput(1)->getDims(), (Shape{1, 2, 3}));
    }
}

TEST(Any, Attr) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    string kernelName = "fake_kernel_name";
    vector<int> attr = {2, 3, 2, 1, 4, 4};
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::Float32);
    Tensor i1 = g->addTensor({2, 2, 3}, DataType::Float32);
    Tensor o0 = g->addTensor({3, 2, 3}, DataType::Float32);
    auto anyOp = g->addOpWithOutputs<AnyObj>(TensorVec{i0, i1}, TensorVec{o0},
                                             kernelName, attr);
    EXPECT_EQ(anyOp->getOpAttrVector(), attr);
}
