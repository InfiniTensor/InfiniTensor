#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/matmul_integer.h"

#include "test.h"

namespace infini {
using ExpectOutput = vector<float>;

TEST(MatmulInteger, ShapeInference) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        auto A = g->addTensor(Shape{1, 4, 2}, DataType::Int8);
        auto B = g->addTensor(Shape{1, 2, 12}, DataType::Int8);
        auto op = g->addOp<MatmulIntegerObj>(A, B, nullptr, nullptr, nullptr);
        auto C = op->getOutputs()[0];
        EXPECT_EQ(C->getDims(), (Shape{1, 4, 12}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        auto A = g->addTensor(Shape{1, 4, 2}, DataType::UInt8);
        auto B = g->addTensor(Shape{1, 2, 12}, DataType::UInt8);
        auto A_Zero = g->addTensor(Shape{1, 4, 1}, DataType::UInt8);
        auto B_Zero = g->addTensor(Shape{1, 1, 12}, DataType::UInt8);
        auto op = g->addOp<MatmulIntegerObj>(A, B, nullptr, A_Zero, B_Zero);
        auto C = op->getOutputs()[0];
        EXPECT_EQ(C->getDims(), (Shape{1, 4, 12}));
    }
}

}; // namespace infini
