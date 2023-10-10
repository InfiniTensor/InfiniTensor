#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
TEST(Unary, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({2}, DataType::Float32);
        auto op = g->addOp<GeluObj>(i0, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2}));
    }
}

} // namespace infini
