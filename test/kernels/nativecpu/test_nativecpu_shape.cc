#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

TEST(Shape, NativeCpu) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 3, 4}, DataType::Float32);
    auto op = g->addOp<ShapeObj>(input, nullptr);
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    // The output holds the dimensions of the input, whatever the input holds.
    auto o = g->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(o->equalData(vector<int64_t>{2, 3, 4}));
}

} // namespace infini
