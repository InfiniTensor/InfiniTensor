#include "core/graph.h"
#include "core/runtime.h"
#include "nnet/Visitor/MatchReshapeVisitor.h"
#include "nnet/expr.h"
#include "nnet/nmutator.h"
#include "nnet/routine.h"
#include "nnet/test.h"
#include "operators/matmul.h"
#include <chrono>
using namespace infini;
using namespace std;

TEST(nnet, MemboundOpInterpretation) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 2, 3}, DataType::UInt32);
    Tensor w0 = g->addTensor({1, 3, 4}, DataType::UInt32);
    Tensor o0 = g->addTensor({1, 2, 4}, DataType::UInt32);
    g->dataMalloc();
    i0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    w0->copyData(vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    g->addOpWithOutputs<MatmulObj>(i0, w0, o0);
    NMutator nmutator(NMutator::Mode::ToNaiveMembound);
    auto mutations = nmutator.run(g);
    ASSERT_EQ(mutations.size(), 1u);
    Graph gNew = mutations[0];
    gNew->print();

    gNew->dataMalloc();
    runtime->run(gNew);
    // check answer
    auto ops = gNew->getOperators();
    EXPECT_EQ(ops.size(), 1u);
    auto membound = ops[0];
    EXPECT_EQ(membound->getOpType(), OpType::MemBound);
    auto ans = make_ref<TensorObj>(Shape{1, 2, 4}, DataType::UInt32, runtime);
    ans->dataMalloc();
    ans->copyData(vector<uint32_t>{38, 44, 50, 56, 83, 98, 113, 128});
    EXPECT_TRUE(membound->getOutput()->equalData(ans));
}