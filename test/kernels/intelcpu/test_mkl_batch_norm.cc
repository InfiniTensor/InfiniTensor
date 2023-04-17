
#include "core/graph.h"
#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/batch_norm.h"
#include "test.h"

namespace infini {
TEST(MklBatchNorm, run) {
    // Runtime
    auto runtime = make_ref<MklRuntimeObj>();

    // Build graph
    Graph g = make_ref<GraphObj>(runtime);
    auto i = g->addTensor(Shape{1, 3, 2, 2}, DataType::Float32);
    auto mean = g->addTensor(Shape{3}, DataType::Float32);
    auto var = g->addTensor(Shape{3}, DataType::Float32);
    auto scale = g->addTensor(Shape{3}, DataType::Float32);
    auto bias = g->addTensor(Shape{3}, DataType::Float32);
    auto op =
        g->addOp<BatchNormObj>(i, nullptr, mean, var, scale, bias, 0.9, 0);
    g->dataMalloc();
    i->setData(IncrementalGenerator());
    mean->copyin(vector<float>{1, 6, 9});
    var->copyin(vector<float>{4, 1, 9});
    scale->setData(OneGenerator());
    bias->setData(ZeroGenerator());

    runtime->run(g);

    auto o = op->getOutput();
    EXPECT_TRUE(o->equalData(vector<float>{
        -0.5, 0, 0.5, 1, -2, -1, 0, 1, -0.3333333, 0, 0.3333333, 0.6666667}));
}

} // namespace infini
