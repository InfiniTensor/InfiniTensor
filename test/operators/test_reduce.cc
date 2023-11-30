#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/reduce.h"

#include "test.h"

namespace infini {

template <typename ReduceObjT> void testShapeInference() {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReduceObjT>(i, nullptr, std::nullopt, true);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 1, 1}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReduceObjT>(i, nullptr, vector<int>{1, 3}, true);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 1, 3, 1}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReduceObjT>(i, nullptr, vector<int>{-3, 3}, true);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 1, 3, 1}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReduceObjT>(i, nullptr, std::nullopt, false);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReduceObjT>(i, nullptr, vector<int>{1, 3}, false);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ReduceObjT>(i, nullptr, vector<int>{-3, 3}, false);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3}));
    }
}

TEST(ReduceMean, ShapeInference) {
    testShapeInference<ReduceMeanObj>();
    testShapeInference<ReduceSumObj>();
}

} // namespace infini
