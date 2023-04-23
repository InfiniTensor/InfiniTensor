
#include "core/graph.h"
#include "core/kernel.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/element_wise.h"
#include "operators/unary.h"
#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;
template <class T>
void testBinary(const std::function<void(void *, size_t, DataType)> &generator,
                const Shape &shape, const ExpectOutput &ansVec) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto a = g->addTensor(shape, DataType::Float32);
    auto b = g->addTensor(shape, DataType::Float32);
    auto op = g->addOp<T>(a, b, nullptr);
    g->dataMalloc();
    a->setData(generator);
    b->setData(generator);

    runtime->run(g);

    auto c = op->getOutput();
    //  check results on CPU
    EXPECT_TRUE(c->equalData(ansVec));
}

TEST(dnnl_Binary, run) {
    testBinary<AddObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                       ExpectOutput{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22});
    testBinary<SubObj>(IncrementalGenerator(), Shape{1, 2, 2, 3},
                       ExpectOutput{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    testBinary<MulObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3},
        ExpectOutput{0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121});

    testBinary<DivObj>(OneGenerator(), Shape{1, 2, 2, 3},
                       ExpectOutput{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
}

TEST(sycl_Pow, run) {
    testBinary<PowObj>(IncrementalGenerator(), Shape{1, 2, 2, 1},
                       ExpectOutput{1, 1, 4, 27});
}

template <class T>
void testUnary(const std::function<void(void *, size_t, DataType)> &generator,
               const Shape &shape) {
    // Runtime
    Runtime rCpu = NativeCpuRuntimeObj::getInstance();
    auto rMkl = make_ref<MklRuntimeObj>();

    // Build input data on CPU

    Graph gCpu = make_ref<GraphObj>(rCpu);
    Tensor iCpu = gCpu->addTensor(shape, DataType::Float32);
    auto opCpu = gCpu->addOp<T>(iCpu, nullptr);
    gCpu->dataMalloc();
    iCpu->setData(generator);
    rCpu->run(gCpu);

    // MKL
    Graph gMkl = make_ref<GraphObj>(rMkl);
    auto iMkl = gMkl->addTensor(shape, DataType::Float32);
    auto opMkl = gMkl->addOp<T>(iMkl, nullptr);
    gMkl->dataMalloc();
    iMkl->setData(generator);
    rMkl->run(gMkl);

    // Check
    EXPECT_TRUE(opCpu->getOutput()->equalData(opMkl->getOutput()));
}

TEST(dnnl_Unary, run) {
    testUnary<ReluObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<SigmoidObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<AbsObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
    testUnary<TanhObj>(IncrementalGenerator(), Shape{1, 2, 2, 3});
}

TEST(dnnl_clip, run) {
    auto rMkl = make_ref<MklRuntimeObj>();

    // MKL
    Graph gMkl = make_ref<GraphObj>(rMkl);
    auto iMkl = gMkl->addTensor(Shape{1, 2, 2, 3}, DataType::Float32);
    auto opMkl = gMkl->addOp<ClipObj>(iMkl, nullptr, 3, 7);
    gMkl->dataMalloc();
    iMkl->copyin(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    rMkl->run(gMkl);

    // Check
    EXPECT_TRUE(opMkl->getOutput()->equalData(
        vector<float>{3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7}));
}

template <class T>
void testBinaryBroadcast(
    const std::function<void(void *, size_t, DataType)> &generator,
    const Shape &shapeA, const Shape &shapeB, const ExpectOutput &ansVec) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto a = g->addTensor(shapeA, DataType::Float32);
    auto b = g->addTensor(shapeB, DataType::Float32);
    auto op = g->addOp<T>(a, b, nullptr);
    g->dataMalloc();
    a->setData(generator);
    b->setData(generator);

    runtime->run(g);

    auto c = op->getOutput();
    //  check results on CPU
    EXPECT_TRUE(c->equalData(ansVec));
}

TEST(dnnl_Binary_broadcast, run) {
    testBinaryBroadcast<AddObj>(
        IncrementalGenerator(), Shape{1, 2, 2, 3}, Shape{2, 1, 2, 1},
        ExpectOutput{0., 1., 2., 4., 5., 6., 6., 7., 8.,  10., 11., 12.,
                     2., 3., 4., 6., 7., 8., 8., 9., 10., 12., 13., 14.});
    testBinaryBroadcast<AddObj>(
        IncrementalGenerator(), Shape{1, 4, 5}, Shape{2, 3, 1, 1},
        ExpectOutput{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10.,
                     11., 12., 13., 14., 15., 16., 17., 18., 19., 1.,  2.,
                     3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13.,
                     14., 15., 16., 17., 18., 19., 20., 2.,  3.,  4.,  5.,
                     6.,  7.,  8.,  9.,  10., 11., 12., 13., 14., 15., 16.,
                     17., 18., 19., 20., 21., 3.,  4.,  5.,  6.,  7.,  8.,
                     9.,  10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
                     20., 21., 22., 4.,  5.,  6.,  7.,  8.,  9.,  10., 11.,
                     12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22.,
                     23., 5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                     15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
}
}; // namespace infini
