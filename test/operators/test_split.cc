#include "core/graph.h"
#include "core/runtime.h"
#include "operators/split.h"

#include "test.h"

namespace infini {
TEST(Split, ShapeInfer) {
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto input = g->addTensor({1, 3, 2, 15}, DataType::Float32);

        auto op = g->addOp<SplitObj>(input, std::nullopt, 3, 4);
        EXPECT_EQ(op->numOutputs(), 4);
        EXPECT_EQ(op->getOutputs().size(), (size_t)4);
        EXPECT_EQ(op->getOutput(0)->getDims(), (Shape{1, 3, 2, 3}));
        EXPECT_EQ(op->getOutput(1)->getDims(), (Shape{1, 3, 2, 3}));
        EXPECT_EQ(op->getOutput(2)->getDims(), (Shape{1, 3, 2, 3}));
        EXPECT_EQ(op->getOutput(3)->getDims(), (Shape{1, 3, 2, 6}));
    }

    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto input = g->addTensor({1, 3, 2, 15}, DataType::Float32);

        auto op =
            g->addOp<SplitObj>(input, std::nullopt, 3, vector<int>{1, 2, 2});
        EXPECT_EQ(op->getOutputs().size(), (size_t)3);
        EXPECT_EQ(op->numOutputs(), 3);
        EXPECT_EQ(op->getOutput(0)->getDims(), (Shape{1, 3, 2, 3}));
        EXPECT_EQ(op->getOutput(1)->getDims(), (Shape{1, 3, 2, 6}));
        EXPECT_EQ(op->getOutput(2)->getDims(), (Shape{1, 3, 2, 6}));
    }
}

} // namespace infini
