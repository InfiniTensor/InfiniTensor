#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

TEST(Transpose, NativeCpu) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    Shape permute = {0, 2, 1, 3};
    auto input = g->addTensor({1, 2, 3, 4}, DataType::Float32);
    auto op = g->addOp<TransposeObj>(input, nullptr, permute);
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(o->equalData(vector<float>{0, 1, 2,  3,  12, 13, 14, 15,
                                           4, 5, 6,  7,  16, 17, 18, 19,
                                           8, 9, 10, 11, 20, 21, 22, 23}));
}

} // namespace infini
