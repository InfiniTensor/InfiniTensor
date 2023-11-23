#include "core/graph.h"
#include "core/runtime.h"
#include "operators/sendrecv.h"
#include "test.h"

namespace infini {
TEST(SendRecv, ShapeTypeInfer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    int source = 0;
    int destination = 1;
    Shape dims = {1, 3, 2, 4};
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor(dims, DataType::Float32);
        auto op =
            g->addOp<SendRecvObj>(input, nullptr, source, destination, dims);
        EXPECT_EQ(op->getOpType(), OpType::SendRecv);
        EXPECT_EQ(op->getOutput()->getDims(), (dims));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
}
} // namespace infini
