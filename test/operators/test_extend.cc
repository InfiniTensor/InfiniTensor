#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/extend.h"

#include "test.h"

namespace infini {

TEST(Extend, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3, 4}, DataType::Float32);
        auto op = g->addOp<ExtendObj>(i, nullptr, 2, 1);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 6, 4}));
    }
}

} // namespace infini
