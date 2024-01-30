#include "core/graph.h"
#include "core/runtime.h"
#include "operators/reshape.h"

#include "test.h"

namespace infini {
TEST(Identity, NativeCpu) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto t1 = g->addTensor({2, 2, 3, 1}, DataType::Float32);
    auto op = g->addOp<IdentityObj>(t1, nullptr);
    g->dataMalloc();
    t1->setData(IncrementalGenerator());

    runtime->run(g);
    EXPECT_TRUE(op->getOutput()->equalData(
        vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
}
} // namespace infini
