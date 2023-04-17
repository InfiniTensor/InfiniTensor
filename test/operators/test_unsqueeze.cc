#include "core/graph.h"
#include "core/runtime.h"
#include "operators/unsqueeze.h"

#include "test.h"

namespace infini {
TEST(Unsqueeze, ShapeInfer) {
    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto input = g->addTensor({1, 3, 2, 15}, DataType::Float32);
        vector<int> index{1, 6, 0};
        auto op = g->addOp<UnsqueezeObj>(input, index, nullptr);

        EXPECT_EQ(op->getOutput(0)->getDims(), (Shape{1, 1, 1, 3, 2, 15, 1}));
    }

    {
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Graph g = make_ref<GraphObj>(runtime);
        auto input = g->addTensor({1, 3, 2, 15}, DataType::Float32);
        vector<int> index{-6, -1, -7};
        auto op = g->addOp<UnsqueezeObj>(input, index, nullptr);

        EXPECT_EQ(op->getOutput(0)->getDims(), (Shape{1, 1, 1, 3, 2, 15, 1}));
    }
}
} // namespace infini
