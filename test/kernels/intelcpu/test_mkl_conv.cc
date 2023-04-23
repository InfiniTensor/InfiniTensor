#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "core/runtime.h"
#include "intelcpu/mkl_runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

void testConvDnnl(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    auto mklRuntime = MklRuntimeObj::getInstance();
    Graph gMkl = make_ref<GraphObj>(mklRuntime);

    Tensor i0 = gMkl->addTensor({1, 3, 4, 4}, DataType::Float32);
    Tensor w0 = gMkl->addTensor({2, 3, 3, 3}, DataType::Float32);

    // Build  graph
    auto conv = gMkl->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 2, 1, 1, 2);
    // Malloc data for all tensors in a graph.
    gMkl->dataMalloc();
    i0->setData(generator);
    w0->setData(generator);

    mklRuntime->run(gMkl);
    EXPECT_TRUE(conv->getOutput(0)->equalData(ansVec));
}

TEST(dnnl_Conv, run) {
    testConvDnnl(OneGenerator(), vector<float>{12, 12, 18, 18, 12, 12, 18, 18});
    testConvDnnl(
        IncrementalGenerator(),
        vector<float>{4794, 4386, 8199, 7506, 11274, 10542, 20835, 19656});
}

TEST(mkl_Conv, groups) {
    auto mklRuntime = MklRuntimeObj::getInstance();
    Graph gMkl = make_ref<GraphObj>(mklRuntime);

    Tensor i0 = gMkl->addTensor({1, 3, 4, 4}, DataType::Float32);
    Tensor w0 = gMkl->addTensor({3, 1, 3, 3}, DataType::Float32);

    // Build  graph
    auto conv = gMkl->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 1, 1, 1, 1);
    // Malloc data for all tensors in a graph.
    gMkl->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());

    mklRuntime->run(gMkl);

    EXPECT_TRUE(conv->getOutput(0)->equalData(vector<float>{
        73,   121,  154,  103,  171,  258,  294,  186,  279,  402,
        438,  270,  139,  187,  202,  113,  1123, 1675, 1762, 1161,
        1710, 2535, 2652, 1737, 2034, 3003, 3120, 2037, 1285, 1885,
        1954, 1267, 3325, 4957, 5098, 3371, 4977, 7404, 7602, 5016,
        5517, 8196, 8394, 5532, 3583, 5311, 5434, 3573}));
}

TEST(mkl_Conv, tune) {
    auto mklRuntime = MklRuntimeObj::getInstance();
    Graph gMkl = make_ref<GraphObj>(mklRuntime);

    Tensor i0 = gMkl->addTensor({1, 3, 224, 224}, DataType::Float32);
    Tensor w0 = gMkl->addTensor({2, 3, 3, 3}, DataType::Float32);
    auto conv = gMkl->addOp<ConvObj>(i0, w0, nullptr, 1, 1, 1, 1, 1, 1);
    gMkl->dataMalloc();

    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());

    // Execute on CUDA
    bool tune = true;
    mklRuntime->run(gMkl, tune);

    // check record
    auto kernelAttrs =
        KernelAttrs{Device::INTELCPU, conv->getOpType(), DataType::Float32};
    auto perfKey = PerfEngine::Key{kernelAttrs, conv->getOpPerfKey()};
    std::optional<PerfRecord> perfData =
        PerfEngine::getInstance().getPerfData(perfKey);
    ASSERT_TRUE(perfData.has_value());
}
} // namespace infini
