#include "core/graph.h"
#include "core/runtime.h"
#include "operators/all_reduce.h"
#include "test.h"

namespace infini {
TEST(AllReuce, ShapeTypeInfer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 3, 2, 4}, DataType::Float32);
        auto op = g->addOp<AllReduceSumObj>(input, nullptr);
        EXPECT_EQ(op->getOpType(), OpType::AllReduceSum);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 4}));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 3, 2, 4}, DataType::Float32);
        auto op = g->addOp<AllReduceProdObj>(input, nullptr);
        EXPECT_EQ(op->getOpType(), OpType::AllReduceProd);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 4}));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 3, 2, 4}, DataType::Float32);
        auto op = g->addOp<AllReduceMinObj>(input, nullptr);
        EXPECT_EQ(op->getOpType(), OpType::AllReduceMin);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 4}));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 3, 2, 4}, DataType::Float32);
        auto op = g->addOp<AllReduceMaxObj>(input, nullptr);
        EXPECT_EQ(op->getOpType(), OpType::AllReduceMax);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 4}));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 3, 2, 4}, DataType::Float32);
        auto op = g->addOp<AllReduceAvgObj>(input, nullptr);
        EXPECT_EQ(op->getOpType(), OpType::AllReduceAvg);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 4}));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
}
} // namespace infini
