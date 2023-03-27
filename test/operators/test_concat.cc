#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "test.h"

namespace infini {
TEST(Concat, ShapeInfer) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto t1 = g->addTensor({1, 3, 2, 4}, DataType::Float32);
    auto t2 = g->addTensor({1, 3, 2, 5}, DataType::Float32);

    auto op = g->addOp<ConcatObj>(TensorVec{t1, t2}, nullptr, 3);
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 9}));
}

} // namespace infini
