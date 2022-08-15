#include "core/graph.h"
#include "core/run_enigne.h"
#include "operators/matmul.h"
#include "test.h"

namespace infini {

TEST(Graph, build_and_run) {
    Graph g = make_ref<GraphObj>();
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::Int32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::Int32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::Int32);
    g->dataMalloc();
    i0->copyData(vector<VType>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data());
    w0->copyData(vector<VType>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data());
    g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    RunEngine(Device::CPU).run(g);
    // check answer
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::Int32);
    ans->dataMalloc();
    ans->copyData(vector<VType>{38, 44, 50, 56, 83, 98, 113, 128}.data());
    EXPECT_TRUE(o0->equalData(ans));
}

TEST(Graph, perf_engine) {
    Graph g = make_ref<GraphObj>();
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::Int32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::Int32);
    auto matmul = g->addOp<MatmulObj>(i0, w0, nullptr);

    g->dataMalloc();
    i0->copyData(vector<VType>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data());
    w0->copyData(vector<VType>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}.data());
    RunEngine(Device::CPU).run(g, true, true);
    double perfTime = RunEngine(Device::CPU).getPerfTime(g);
    // The example matmul takes 0.0036ms with one core
    EXPECT_GT(perfTime, 0);
    EXPECT_LT(perfTime, 0.01);
    // check answer
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::Int32);
    ans->dataMalloc();
    ans->copyData(vector<VType>{38, 44, 50, 56, 83, 98, 113, 128}.data());
    EXPECT_TRUE(matmul->getOutput()->equalData(ans));
}

} // namespace infini