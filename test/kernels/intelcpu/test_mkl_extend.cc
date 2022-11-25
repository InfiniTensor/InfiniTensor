#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/extend.h"

#include "test.h"

namespace infini {

TEST(MKL_Extend, run) {
    Runtime runtime = MklRuntimeObj::getInstance();

    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor(Shape{2, 3, 2, 2}, DataType::Float32);
    auto op = g->addOp<ExtendObj>(i, nullptr, 1, 1);
    g->dataMalloc();
    i->setData(IncrementalGenerator());

    // Execute
    runtime->run(g);

    auto o = op->getOutput();

    //  check results on CPU
    EXPECT_TRUE(o->equalData(vector<float>{
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 0,  1,  2,  3,
        4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}));
}
} // namespace infini
