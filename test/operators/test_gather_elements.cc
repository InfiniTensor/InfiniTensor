#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/gather.h"

#include "test.h"

namespace infini {

TEST(Gather, ShapeTypeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({3, 3, 3}, DataType::Int32);
        Tensor index = g->addTensor({2, 3, 3}, DataType::Int32);
        auto op = g->addOp<GatherElementsObj>(i, index, nullptr, 0);
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Int32);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 3}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 4, 2}, DataType::Float32);
        Tensor index = g->addTensor({2, 1, 2}, DataType::Int64);
        auto op = g->addOp<GatherElementsObj>(i, index, nullptr, 1);
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 1, 2}));
    }
}
} // namespace infini
