#include "core/graph.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/split.h"

#include "test.h"

namespace infini {

TEST(Split, Mkl) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 10, 2, 1}, DataType::Float32);
    auto op = g->addOp<SplitObj>(input, std::nullopt, 1, 3);
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    EXPECT_EQ(op->getOutputs().size(), (size_t)3);
    auto o0 = g->cloneTensor(op->getOutput(0));
    auto o1 = g->cloneTensor(op->getOutput(1));
    auto o2 = g->cloneTensor(op->getOutput(2));
    EXPECT_TRUE(
        o0->equalData(vector<float>{0, 1, 2, 3, 4, 5, 20, 21, 22, 23, 24, 25}));
    EXPECT_TRUE(o1->equalData(
        vector<float>{6, 7, 8, 9, 10, 11, 26, 27, 28, 29, 30, 31}));
    EXPECT_TRUE(o2->equalData(vector<float>{12, 13, 14, 15, 16, 17, 18, 19, 32,
                                            33, 34, 35, 36, 37, 38, 39}));
}

} // namespace infini
