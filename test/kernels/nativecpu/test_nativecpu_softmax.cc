#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/softmax.h"

#include "test.h"

namespace infini {

// Softmax normalises along one axis. Summing every element of the tensor
// instead answers a single distribution spread over the whole thing, which
// still sums to one overall and so looks unremarkable until a row is checked
// on its own.
TEST(Softmax, NativeCpuAlongLastAxis) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 2}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(input, nullptr, 1);
    g->dataMalloc();
    // Each row is the same pair, so each row must come back the same, and the
    // gap of one between the two entries fixes what the pair has to be.
    input->copyin(vector<float>{0, 1, 5, 6});

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    float lo = 1 / (1 + std::exp(1.0f));
    float hi = 1 - lo;
    EXPECT_TRUE(o->equalData(vector<float>{lo, hi, lo, hi}));
}

TEST(Softmax, NativeCpuAlongFirstAxis) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 2}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(input, nullptr, 0);
    g->dataMalloc();
    // The same numbers as above. Normalising down the columns rather than
    // across the rows has to give a different answer, which is what tells the
    // two axes apart.
    input->copyin(vector<float>{0, 1, 5, 6});

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    float lo = 1 / (1 + std::exp(5.0f));
    float hi = 1 - lo;
    EXPECT_TRUE(o->equalData(vector<float>{lo, lo, hi, hi}));
}

// A middle axis is the case where neither the runs nor their elements are
// next to each other in memory, so a wrong stride cannot hide.
TEST(Softmax, NativeCpuAlongMiddleAxis) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({2, 2, 2}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(input, nullptr, 1);
    g->dataMalloc();
    input->copyin(vector<float>{0, 0, 1, 1, 3, 3, 4, 4});

    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    float lo = 1 / (1 + std::exp(1.0f));
    float hi = 1 - lo;
    EXPECT_TRUE(o->equalData(vector<float>{lo, lo, hi, hi, lo, lo, hi, hi}));
}

// Exponentiating these directly overflows to infinity, and every entry then
// divides infinity by infinity. The answer is an ordinary pair of fractions.
//
// The values are read out and checked here rather than handed to `equalData`,
// because a comparison against not-a-number is false whichever way it is
// asked: `equalData` finds no difference greater than its tolerance and
// reports a match. Only asking outright whether the number is finite catches
// this.
TEST(Softmax, NativeCpuLargeValuesDoNotOverflow) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor({1, 2}, DataType::Float32);
    auto op = g->addOp<SoftmaxObj>(input, nullptr, 1);
    g->dataMalloc();
    input->copyin(vector<float>{1000, 1001});

    runtime->run(g);

    auto got = g->cloneTensor(op->getOutput(0))->copyout<float>();
    ASSERT_EQ(got.size(), 2u);
    EXPECT_TRUE(std::isfinite(got[0])) << got[0];
    EXPECT_TRUE(std::isfinite(got[1])) << got[1];
    float lo = 1 / (1 + std::exp(1.0f));
    EXPECT_NEAR(got[0], lo, 1e-6);
    EXPECT_NEAR(got[1], 1 - lo, 1e-6);
}

} // namespace infini
