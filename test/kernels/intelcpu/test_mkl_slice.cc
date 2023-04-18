#include "core/graph.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/slice.h"
#include "test.h"

namespace infini {
TEST(MKL_Slice, run) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // Build input data
    Tensor i = g->addTensor(Shape{3, 2, 1, 5}, DataType::Float32);
    auto op =
        g->addOp<SliceObj>(i, nullptr, vector<int>{1, 1}, vector<int>{2, 5},
                           vector<int>{0, 3}, std::nullopt);
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute
    runtime->run(g);

    auto o = op->getOutput();
    EXPECT_TRUE(o->equalData(vector<float>{11, 12, 13, 14, 16, 17, 18, 19}));
}
} // namespace infini
