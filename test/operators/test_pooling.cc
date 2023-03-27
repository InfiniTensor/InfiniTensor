#include "core/graph.h"
#include "core/runtime.h"
#include "operators/pooling.h"
#include "test.h"

namespace infini {
using KDPS = vector<int>;
using ExpectOutput = vector<float>;
TEST(MaxPool, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 64, 162, 162}, DataType::UInt32);
        const int kh = 3, kw = 3, dh = 1, dw = 1, ph = 0, pw = 0, sh = 2,
                  sw = 2;
        auto op =
            g->addOp<MaxPoolObj>(i, nullptr, kh, kw, dh, dw, ph, pw, sh, sw);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 64, 80, 80}));
    }

    { // dilation & stride
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 64, 162, 162}, DataType::UInt32);
        auto op = g->addOp<MaxPoolObj>(i, nullptr, 4, 3, 1, 1, 2, 1, 1, 2);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 64, 163, 81}));
    }
}

TEST(MaxPool, NaiveCPU) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(cpuRuntime);
    Tensor i = g->addTensor({1, 2, 5, 5}, DataType::UInt32);
    auto op = g->addOp<MaxPoolObj>(i, nullptr, 3, 3, 1, 1, 1, 1, 2, 2);

    g->dataMalloc();
    i->setData(IncrementalGenerator());
    cpuRuntime->run(g, true, true);
    double perfTime = cpuRuntime->getPerfTime(g);
    // The example matmul takes 0.0036ms with one core
    EXPECT_GT(perfTime, 0);
    EXPECT_LT(perfTime, 5);
    // check answer
    vector<uint32_t> ans = {6,  8,  9,  16, 18, 19, 21, 23, 24,
                            31, 33, 34, 41, 43, 44, 46, 48, 49};
    EXPECT_TRUE(op->getOutput()->equalData(ans));
}

TEST(AvgPool, NaiveCPU) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(cpuRuntime);
    Tensor i = g->addTensor({1, 2, 5, 5}, DataType::Float32);
    auto op = g->addOp<AvgPoolObj>(i, nullptr, 3, 3, 1, 1, 1, 1, 2, 2);

    g->dataMalloc();
    i->setData(IncrementalGenerator());
    cpuRuntime->run(g, true, true);

    // check answer
    vector<float> ans = {
        1.33333337, 3.0000,  2.66666675, 7.0000,     12.0000,   9.0000,
        8.0000,     13.0000, 9.33333302, 12.444447,  19.666666, 13.7777777,
        23.666666,  37.0000, 25.666666,  19.1111107, 29.666666, 20.4444447};
    EXPECT_TRUE(op->getOutput()->equalData(ans));

    double perfTime = cpuRuntime->getPerfTime(g);
    // The example matmul takes 0.0036ms with one core
    EXPECT_GT(perfTime, 0);
    EXPECT_LT(perfTime, 5);
}

} // namespace infini
