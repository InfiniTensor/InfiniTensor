#include "core/graph.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/unsqueeze.h"

#include "test.h"

namespace infini {

TEST(Unsqueeze, Mkl) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 3, 3}, DataType::Float32);
    vector<int> index{1, 0};
    auto op = g->addOp<UnsqueezeObj>(input, index, nullptr);
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    //  check results
    EXPECT_TRUE(o->equalData(input));
}
} // namespace infini
