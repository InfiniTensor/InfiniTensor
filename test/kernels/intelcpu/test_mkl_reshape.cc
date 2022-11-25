#include "core/graph.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/reshape.h"

#include "test.h"

namespace infini {

TEST(Reshape, Mkl) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 3, 3, 4}, DataType::Float32);
    auto op = g->addOp<ReshapeObj>(input, nullptr, Shape{3, 2, 4, 3});
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    //  check results
    EXPECT_TRUE(o->equalData(input));
}

TEST(Flatten, Mkl) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 3, 3, 4}, DataType::Float32);
    auto op = g->addOp<FlattenObj>(input, nullptr, 2);
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    //  check results
    EXPECT_TRUE(o->equalData(input));
}

TEST(Identify, Mkl) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 3, 3, 4}, DataType::Float32);
    auto op = g->addOp<IdentityObj>(input, nullptr);
    g->dataMalloc();
    input->setData(IncrementalGenerator());

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    //  check results
    EXPECT_TRUE(o->equalData(input));
}
} // namespace infini
