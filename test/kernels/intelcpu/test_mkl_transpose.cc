#include "core/graph.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/transpose.h"

#include "test.h"

namespace infini {

TEST(Transpose, Mkl) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 3, 1, 2}, DataType::Float32);
    auto op = g->addOp<TransposeObj>(input, nullptr, Shape{0, 2, 3, 1});
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    //  check results
    EXPECT_TRUE(o->equalData(
        vector<float>{0., 2., 4., 1., 3., 5., 6., 8., 10., 7., 9., 11.}));
}

} // namespace infini
