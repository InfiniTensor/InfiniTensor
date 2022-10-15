#include "core/graph.h"
#include "core/runtime.h"
#include "operators/slice.h"
#include "test.h"

namespace infini {
TEST(Slice, ShapeInference) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({10, 64, 162, 162}, DataType::UInt32);
        auto op = g->addOp<SliceObj>(i, nullptr, vector<int>{2, 10, 1, 5},
                                     vector<int>{3, 10, 100, 100}, std::nullopt,
                                     std::nullopt);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 1, 100, 96}));
    }
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({10, 64, 162, 162}, DataType::UInt32);
        auto op = g->addOp<SliceObj>(i, nullptr, vector<int>{2, 5},
                                     vector<int>{3, 100}, vector<int>{1, 3},
                                     std::nullopt);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{10, 2, 162, 96}));
    }
}

} // namespace infini
