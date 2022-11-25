#include "core/graph.h"
#include "core/runtime.h"
#include "operators/pad.h"
#include "test.h"

namespace infini {
TEST(Pad, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 64, 162, 162}, DataType::UInt32);
        auto op = g->addOp<PadObj>(
            i, nullptr, vector<int>{2, 10, 1, 5, 0, 10, 1, 5}, std::nullopt);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 84, 164, 172}));
    }
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 64, 162, 162}, DataType::UInt32);
        auto op = g->addOp<PadObj>(i, nullptr, vector<int>{2, 10, 1, 5},
                                   vector<int>{0, 3});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{4, 64, 162, 177}));
    }
}

} // namespace infini
