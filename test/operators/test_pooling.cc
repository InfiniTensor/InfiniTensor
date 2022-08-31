#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/pooling.h"
#include "test.h"

namespace infini {
using KDPS = vector<int>;
using ExpectOutput = vector<float>;
TEST(MaxPool, ShapeInference) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
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
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
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
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
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

template <class T>
void testPoolCudnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const KDPS &kdps, const ExpectOutput &ansVec) {
    EXPECT_TRUE(kdps.size() == 8);
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor i0cpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    i0cpu->dataMalloc();
    i0cpu->setData(generator);

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto i0 = g->cloneTensor(i0cpu);
    auto pool = g->addOp<T>(i0, nullptr, kdps[0], kdps[1], kdps[2], kdps[3],
                            kdps[4], kdps[5], kdps[6], kdps[7]);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto o0 = pool->getOutput();
    auto cpuo0 = o0->clone(cpuRuntime);

    // check results on CPU
    EXPECT_TRUE(cpuo0->equalData(ansVec));
}

TEST(MaxPool, CuDNN) {
    testPoolCudnn<MaxPoolObj>(IncrementalGenerator(), Shape{1, 2, 5, 5},
                              KDPS{3, 3, 1, 1, 1, 1, 2, 2},
                              ExpectOutput{6, 8, 9, 16, 18, 19, 21, 23, 24, 31,
                                           33, 34, 41, 43, 44, 46, 48, 49});
}

TEST(AvgPool, CuDNN) {
    testPoolCudnn<AvgPoolObj>(
        IncrementalGenerator(), Shape{1, 2, 5, 5}, KDPS{3, 3, 1, 1, 1, 1, 2, 2},
        ExpectOutput{1.333333, 3.0000, 2.666667, 7.0000, 12.0000, 9.0000,
                     8.0000, 13.0000, 9.333333, 12.44444, 19.666667, 13.777778,
                     23.666667, 37.0000, 25.666667, 19.111111, 29.666667,
                     20.444444});
}

} // namespace infini