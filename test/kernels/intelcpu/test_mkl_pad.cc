#include "core/graph.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/pad.h"
#include "test.h"

namespace infini {
TEST(Pad, Mkl) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // Build input data
    Tensor i = g->addTensor(Shape{1, 2, 3, 2}, DataType::Float32);
    auto op = g->addOp<PadObj>(i, nullptr, vector<int>{1, 0, 1, 1},
                               vector<int>{0, 3});
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute
    runtime->run(g);

    auto o = op->getOutput();

    //  check results
    EXPECT_TRUE(o->equalData(
        vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,
                      0, 1, 0, 2, 3, 0, 4, 5, 0, 6, 7, 0, 8, 9, 0, 10, 11, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0}));
}
} // namespace infini
