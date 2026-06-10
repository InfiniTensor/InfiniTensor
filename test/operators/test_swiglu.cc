#include "core/graph.h"
#include "core/runtime.h"
#include "operators/swiglu.h"
#include "test.h"

namespace infini {
TEST(SwiGLU, ShapeInference) {
    Runtime runtime = make_ref<RuntimeObj>(Device(Device::Type::kCpu));
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({2, 3, 4}, DataType::Float32);
    Tensor i1 = g->addTensor({2, 3, 4}, DataType::Float32);
    auto op = g->addOp<SwiGLUObj>(i0, i1, nullptr);
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4}));
}
} // namespace infini
