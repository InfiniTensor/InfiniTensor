#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/where.h"

#include "test.h"

namespace infini {

TEST(Where, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor x = g->addTensor({2, 2}, DataType::Float32);
        Tensor y = g->addTensor({2, 2}, DataType::Float32);
        Tensor con = g->addTensor({2, 2}, DataType::Bool);
        auto op = g->addOp<WhereObj>(x, y, con, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 2}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor x = g->addTensor({1, 12, 224, 224}, DataType::Float32);
        Tensor y = g->addTensor({1, 1, 224, 224}, DataType::Float32);
        Tensor con = g->addTensor({1, 224, 1}, DataType::Bool);
        auto op = g->addOp<WhereObj>(x, y, con, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 12, 224, 224}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor x = g->addTensor({12, 224, 224}, DataType::Float32);
        Tensor y = g->addTensor({1, 1, 224, 224}, DataType::Float32);
        Tensor con = g->addTensor({1, 224}, DataType::Bool);
        auto op = g->addOp<WhereObj>(x, y, con, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 12, 224, 224}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor x = g->addTensor({12, 224, 224}, DataType::Float32);
        Tensor y = g->addTensor({1, 1, 224, 224}, DataType::Float32);
        Tensor con = g->addTensor({2, 1, 1, 1, 224}, DataType::Bool);
        auto op = g->addOp<WhereObj>(x, y, con, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 12, 224, 224}));
    }
}

} // namespace infini
