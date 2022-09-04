#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/matmul.h"
#include "test.h"

namespace infini {

TEST(Graph, build_and_run) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    runtime->run(g);
    // check answer
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyData(vector<uint32_t>{38, 44, 50, 56, 83, 98, 113, 128});
    EXPECT_TRUE(o0->equalData(ans));
}

TEST(Graph, perf_engine) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    auto matmul = g->addOp<MatmulObj>(i0, w0, nullptr);

    g->dataMalloc();
    i0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    runtime->run(g, true, true);
    double perfTime = runtime->getPerfTime(g);
    // The example matmul takes 0.0036ms with one core
    EXPECT_GT(perfTime, 0);
    EXPECT_LT(perfTime, 0.01);
    // check answer
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyData(vector<uint32_t>{38, 44, 50, 56, 83, 98, 113, 128});
    EXPECT_TRUE(matmul->getOutput()->equalData(ans));
}

} // namespace infini
