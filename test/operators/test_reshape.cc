#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/reshape.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"

#include "test.h"

namespace infini {

TEST(Reshape, RuntimeTargetTensor) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 3}, DataType::Float32);
    auto target = graph->addTensor({2}, DataType::Int64);
    target->setWeight();
    target->dataMalloc();
    target->copyin(vector<int64_t>{3, 2});
    auto op = graph->addOp<ReshapeObj>(input, target, nullptr);
    EXPECT_EQ(op->numInputs(), 2);
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 2}));
    target->copyin(vector<int64_t>{1, -1});
    graph->shape_infer();
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 6}));
    target->copyin(vector<int64_t>{-1, -1});
    EXPECT_THROW(graph->shape_infer(), std::invalid_argument);
    target->copyin(vector<int64_t>{4, -1});
    EXPECT_THROW(graph->shape_infer(), std::invalid_argument);
    target->copyin(vector<int64_t>{0, -1});
    graph->shape_infer();
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3}));
    graph->dataMalloc();
    input->copyin(vector<float>{1, 2, 3, 4, 5, 6});
    target->copyin(vector<int64_t>{3, 2});
    graph->shape_infer();
    graph->dataMalloc();
    runtime->run(graph);
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 2}));
    EXPECT_EQ(op->getOutput()->copyout<float>(), (vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(Reshape, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReshapeObj>(i, nullptr, Shape{3, 2, 4, 3});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 2, 4, 3}));
    }
}
TEST(Flatten, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<FlattenObj>(i, nullptr, 1);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 36}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<FlattenObj>(i, nullptr, 0);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 72}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<FlattenObj>(i, nullptr, -1);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{18, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<FlattenObj>(i, nullptr, -2);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{6, 12}));
    }
}

TEST(Identity, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<IdentityObj>(i, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 3, 4}));
    }
}

TEST(Squeeze, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 1, 4}, DataType::Float32);
        auto op = g->addOp<SqueezeObj>(i, nullptr, Shape{-2});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 1, 3, 4}, DataType::Float32);
        auto op = g->addOp<SqueezeObj>(i, nullptr, Shape{});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 4}));
    }
}

TEST(Unsqueeze, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 4}, DataType::Float32);
        auto op = g->addOp<UnsqueezeObj>(i, nullptr, Shape{0, 1});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 2, 3, 4}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 4}, DataType::Float32);
        auto op = g->addOp<UnsqueezeObj>(i, nullptr, Shape{-1, -2});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4, 1, 1}));
    }
}

} // namespace infini
