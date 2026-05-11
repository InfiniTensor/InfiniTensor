#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/det.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
TEST(Det, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 3}, DataType::Float32);
        auto op = g->addOp<DetObj>(i, nullptr, std::string("normal"));
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({3, 3}, DataType::Float32);
        auto op = g->addOp<DetObj>(i, nullptr, std::string("logDet"));
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1}));
    }
}

} // namespace infini
