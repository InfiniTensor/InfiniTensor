#include "core/graph.h"
#include "core/runtime.h"
#include "operators/recv.h"
#include "operators/send.h"
#include "test.h"

namespace infini {
TEST(Send, ShapeTypeInfer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    int source = 0;
    int destination = 1;
    Shape dims = {1, 3, 2, 4};
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor(dims, DataType::Float32);
        auto op = g->addOp<SendObj>(input, source, destination, dims, nullptr);
        EXPECT_EQ(op->getOpType(), OpType::Send);
        EXPECT_EQ(op->getInputs(0)->getDims(), (dims));
        EXPECT_EQ(op->getInputs(0)->getDType(), DataType::Float32);
    }
}
TEST(Recv, ShapeTypeInfer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    int source = 0;
    int destination = 1;
    Shape dims = {1, 3, 2, 4};
    int outputType = 1;
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor(dims, DataType::Float32);
        auto op = g->addOp<RecvObj>(nullptr, source, destination, dims,
                                    outputType, input);
        EXPECT_EQ(op->getOpType(), OpType::Recv);
        EXPECT_EQ(op->getOutput()->getDims(), (dims));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
}
} // namespace infini
