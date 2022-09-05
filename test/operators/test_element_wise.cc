#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/element_wise.h"

#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
TEST(ElementWise, ShapeInference) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i0 = g->addTensor({2, 3, 3, 4}, DataType::UInt32);
        Tensor i1 = g->addTensor({2, 3, 3, 4}, DataType::UInt32);
        auto op = g->addOp<AddObj>(i0, i1, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 3, 4}));
    }
}
/*
template <typename T>
void test_element_wise(
    const std::function<void(void *, size_t, DataType)> &generator,
    const vector<uint32_t> &ans) {
    Runtime runtime = CpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 3, 2, 2}, DataType::UInt32);
    Tensor i1 = g->addTensor({2, 3, 1, 2}, DataType::UInt32);
    auto op = g->addOp<T>(i0, i1, nullptr);

    g->dataMalloc();
    i0->setData(generator);
    i1->setData(generator);
    runtime->run(g, true, true);
    // check answer
    EXPECT_TRUE(op->getOutput()->equalData(ans));
}

TEST(ElementWise, NaiveCPU) {
    test_element_wise<AddObj>(IncrementalGenerator(),
                              vector<uint32_t>{0,  2,  2,  4,  6,  8,  8,  10,
                                               12, 14, 14, 16, 6,  8,  8,  10,
                                               12, 14, 14, 16, 18, 20, 20, 22});
    test_element_wise<SubObj>(
        IncrementalGenerator(),
        vector<uint32_t>{0,          0,          2,          2,
                         2,          2,          4,          4,
                         4,          4,          6,          6,
                         4294967290, 4294967290, 4294967292, 4294967292,
                         4294967292, 4294967292, 4294967294, 4294967294,
                         4294967294, 4294967294, 0,          0});
    test_element_wise<MulObj>(
        IncrementalGenerator(),
        vector<uint32_t>{0, 1, 0,  3,  8,  15, 12, 21, 32, 45, 40,  55,
                         0, 7, 12, 21, 32, 45, 48, 63, 80, 99, 100, 121});
    test_element_wise<DivObj>(OneGenerator(),
                              vector<uint32_t>{
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              });
}
*/

template <class T>
void testElementWiseCudnn(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shape, const ExpectOutput &ansVec) {
    Runtime cpuRuntime = CpuRuntimeObj::getInstance();
    auto cudaRuntime = make_ref<CudaRuntimeObj>();

    // Build input data on CPU
    Tensor acpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    acpu->dataMalloc();
    acpu->setData(generator);

    Tensor bcpu = make_ref<TensorObj>(shape, DataType::Float32, cpuRuntime);
    bcpu->dataMalloc();
    bcpu->setData(generator);

    // Build CUDA graph
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto a = g->cloneTensor(acpu);
    auto b = g->cloneTensor(bcpu);
    auto op = g->addOp<T>(a, b, nullptr);

    // allocate CUDA memory
    g->dataMalloc();

    // Execute on CUDA
    cudaRuntime->run(g);

    // clone CUDA output to CPU
    auto c = op->getOutput();
    auto ccpu = c->clone(cpuRuntime);
    // cudaPrintTensor(c);
    //  check results on CPU
    EXPECT_TRUE(ccpu->equalData(ansVec));
}

TEST(ElementWise, CuDNN) {
    testElementWiseCudnn<AddObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});
    testElementWiseCudnn<SubObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    testElementWiseCudnn<MulObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121});

    testElementWiseCudnn<DivObj>(
        OneGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

    testElementWiseCudnn<PowObj>(IncrementalGenerator(), Shape{1, 2, 2, 1},
                                 ExpectOutput{1, 1, 4, 27});
}

} // namespace infini