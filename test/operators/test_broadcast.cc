#include "core/graph.h"
#include "core/runtime.h"
#include "operators/broadcast.h"
#include "test.h"

namespace infini {
TEST(Broadcast, ShapeTypeInfer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    int root = 0;
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor input = g->addTensor({1, 3, 2, 4}, DataType::Float32);
        auto op = g->addOp<BroadcastObj>(input, nullptr, root);
        EXPECT_EQ(op->getOpType(), OpType::Broadcast);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 4}));
        EXPECT_EQ(op->getOutput()->getDType(), DataType::Float32);
    }
}
} // namespace infini
