
#include "core/graph.h"
#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/softmax.h"
#include "test.h"

namespace infini {
TEST(MklSoftmax, run) {
    // Runtime
    auto runtime = make_ref<MklRuntimeObj>();

    // Build input data on intelcpu
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor(Shape{2, 4}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(i, nullptr, 1);
    g->dataMalloc();
    i->copyin(vector<float>{0, 1, 2, 3, 10000, 10001, 10002, 10003});
    runtime->run(g);

    // Check
    EXPECT_TRUE(op->getOutput(0)->equalData(
        vector<float>{0.032058604, 0.08714432, 0.23688284, 0.6439143,
                      0.032058604, 0.08714432, 0.23688284, 0.6439143}));
}

TEST(MklSoftmax, run_axis1) {
    // Runtime
    auto runtime = make_ref<MklRuntimeObj>();

    // Build input data on intelcpu
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor(Shape{2, 2, 2, 2}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(i, nullptr, 1);
    g->dataMalloc();
    i->setData(IncrementalGenerator());
    runtime->run(g);

    // Check
    EXPECT_TRUE(op->getOutput(0)->equalData(vector<float>{
        0.0179862, 0.0179862, 0.0179862, 0.0179862, 0.9820138, 0.9820138,
        0.9820138, 0.9820138, 0.0179862, 0.0179862, 0.0179862, 0.0179862,
        0.9820138, 0.9820138, 0.9820138, 0.9820138}));
}

TEST(MklSoftmax, run_axis2) {
    // Runtime
    auto runtime = make_ref<MklRuntimeObj>();

    // Build input data on intelcpu
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor(Shape{2, 2, 2, 2}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(i, nullptr, 2);
    g->dataMalloc();
    i->setData(IncrementalGenerator());
    runtime->run(g);

    // Check
    EXPECT_TRUE(op->getOutput(0)->equalData(vector<float>{
        0.119203, 0.119203, 0.880797, 0.880797, 0.119203, 0.119203, 0.880797,
        0.880797, 0.119203, 0.119203, 0.880797, 0.880797, 0.119203, 0.119203,
        0.880797, 0.880797}));
}

TEST(MklSoftmax, run_axis3) {
    // Runtime
    auto runtime = make_ref<MklRuntimeObj>();

    // Build input data on intelcpu
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor(Shape{2, 2, 2, 2}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(i, nullptr, 3);
    g->dataMalloc();
    i->setData(IncrementalGenerator());
    runtime->run(g);

    // Check
    EXPECT_TRUE(op->getOutput(0)->equalData(vector<float>{
        0.2689414, 0.7310585, 0.2689414, 0.7310585, 0.2689414, 0.7310585,
        0.2689414, 0.7310585, 0.2689414, 0.7310585, 0.2689414, 0.7310585,
        0.2689414, 0.7310585, 0.2689414, 0.7310585}));
}
} // namespace infini
