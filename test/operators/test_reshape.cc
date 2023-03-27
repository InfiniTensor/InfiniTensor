#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/reshape.h"

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
        auto op = g->addOp<FlattenObj>(i, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{72}));
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

} // namespace infini
