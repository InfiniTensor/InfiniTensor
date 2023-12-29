#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/reshape.h"
#include "operators/squeeze.h"
#include "operators/unsqueeze.h"

#include "test.h"

namespace infini {

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
