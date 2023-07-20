#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/expand.h"

#include "test.h"

namespace infini {

TEST(Expand, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({3, 1}, DataType::Float32);
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{2, 1, 6});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 6}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({3, 1}, DataType::Float32);
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{3, 4});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 4}));
    }
}

} // namespace infini
