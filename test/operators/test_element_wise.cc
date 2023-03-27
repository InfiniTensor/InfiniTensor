#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
TEST(ElementWise, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({2, 3, 3, 4}, DataType::UInt32);
        Tensor i1 = g->addTensor({2, 3, 3, 4}, DataType::UInt32);
        auto op = g->addOp<AddObj>(i0, i1, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 3, 4}));
    }
}

} // namespace infini
