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
/// ONNX spells a target dimension of zero the same way it spells keeping the
/// dimension the input already has in that position, so a zero is read as the
/// latter. That reading needs a position to read from.
TEST(Reshape, AZeroKeepsTheInputDimension) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    { // Within reach, which is what an exporter relies on when it writes a
      // target it does not want to spell out in full.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 4, 6}, DataType::Float32);
        auto op = g->addOp<ReshapeObj>(i, nullptr, Shape{0, 24});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 24}));
    }

    { // Past the end of the input, so the position it would be kept from is
      // not there. Reading it anyway took whatever the memory held and gave a
      // dimension that varied from one run of the same graph to the next.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 4, 6}, DataType::Float32);
        EXPECT_THROW(g->addOp<ReshapeObj>(i, nullptr, Shape{1, 4, 6, 0}),
                     Exception);
    }
}

/// The leftover a target asks for with a minus one is what the other dimensions
/// do not account for, which has no answer when they account for nothing: every
/// size at all divides into no elements the same number of times.
TEST(Reshape, ALeftoverBesideAZeroIsRefused) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor({0, 6}, DataType::Float32);
    EXPECT_THROW(g->addOp<ReshapeObj>(i, nullptr, Shape{0, -1}), Exception);
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
