#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/gather.h"

#include "test.h"

namespace infini {

TEST(Gather, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();

    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor({1, 3, 4, 4}, DataType::UInt32);
    Tensor index = g->addTensor({2, 1, 2}, DataType::UInt32);
    auto op = g->addOp<GatherObj>(i, index, nullptr, 1);
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 2, 1, 2, 4, 4}));
}
} // namespace infini
