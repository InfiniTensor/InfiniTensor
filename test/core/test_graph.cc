#include "core/blob.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "core/search_engine.h"
#include "operators/matmul.h"
#include "operators/unary.h"
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
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->print();
    // check inputOf and outputsOf for tensor
    EXPECT_EQ(i0->getInputOf().size(), 1u);
    EXPECT_EQ(w0->getInputOf().size(), 1u);
    EXPECT_EQ(o0->getInputOf().size(), 0u);
    EXPECT_EQ(i0->getOutputOf(), nullptr);
    EXPECT_EQ(w0->getOutputOf(), nullptr);
    EXPECT_NE(o0->getOutputOf(), nullptr);
    EXPECT_EQ(matmul->getPredecessors().size(), 0u);
    EXPECT_EQ(matmul->getSuccessors().size(), 0u);

    runtime->run(g);
    // check execution results
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

TEST(Graph, test_tensor_id) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto i1 = g->addTensor(i0->clone());
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->print();
    EXPECT_NE(i0->getGuid(), i1->getGuid());
    EXPECT_EQ(i0->getFuid(), i1->getFuid());
    EXPECT_NE(i0->getDataBlob(), nullptr);
    EXPECT_EQ(i1->getDataBlob(), nullptr);
}

TEST(Graph, test_OpVec_ctor) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto o1 = g->addTensor(o0->clone());
    auto matmul = g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    g->addOp<ReluObj>(o1, nullptr);
    g->print();
    puts("=========");
    OpVec ops = g->getOperators();
    Graph g2 = make_ref<GraphObj>(runtime, ops);
    g2->print();
    // Check if the two tensors with the same FUID (o0,o1) remain only one in g2
    EXPECT_EQ(g2->getTensors().size(), 4u);
    EXPECT_EQ(g2->getOperators().size(), 2u);
    map<pair<int, int>, int> inputOutput2Cnt = {
        {{1, 0}, 2}, {{1, 1}, 1}, {{0, 1}, 1}};
    for (auto t : g2->getTensors()) {
        pair<int, int> key = {t->getInputOf().size(),
                              t->getOutputOf() != nullptr};
        EXPECT_GE(inputOutput2Cnt[key], 0);
        inputOutput2Cnt[key]--;
    }
    for (auto [u, v] : inputOutput2Cnt) {
        EXPECT_EQ(v, 0);
    }
}

} // namespace infini
