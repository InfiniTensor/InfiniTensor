#include "core/graph.h"
#include "core/kernel.h"
#include "core/perf_engine.h"
#include "mkl/mkl_runtime.h"
#include "operators/conv.h"

#include "test.h"

namespace infini {

void testConvTransposedMkl(
    const std::function<void(void *, size_t, DataType)> &generator,
    vector<float> ansVec) {
    const auto &[N, C, H, W, F, R, S] = tuple{1, 1, 2, 2, 1, 4, 4};
    const int stride = 1, padding = 0, dilation = 1;

    Runtime runtime = MklRuntimeObj::getInstance();
    Graph gMkl = make_ref<GraphObj>(runtime);
    // Set input data on CPU in a CPU Graph
    Tensor i0 = gMkl->addTensor({N, F, H, H}, DataType::Float32);
    Tensor w0 = gMkl->addTensor({F, C, R, S}, DataType::Float32);
    auto conv = gMkl->addOp<ConvTransposed2dObj>(
        i0, w0, nullptr, padding, padding, stride, stride, dilation, dilation);

    gMkl->dataMalloc();
    i0->setData(generator);
    w0->setData(generator);

    runtime->prepareAndRun(gMkl);
    EXPECT_TRUE(conv->getOutput()->equalData(ansVec));
}

TEST(mkl_ConvTransposed, run) {
    testConvTransposedMkl(IncrementalGenerator(),
                          vector<float>{0.,  0.,  1.,  2.,  3.,  0.,  6.,
                                        12., 18., 16., 8.,  30., 36., 42.,
                                        32., 16., 54., 60., 66., 48., 24.,
                                        62., 67., 72., 45.});
}

TEST(mkl_ConvTransposed, run1) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph gMkl = make_ref<GraphObj>(runtime);
    // Set input data on CPU in a CPU Graph
    Tensor i0 = gMkl->addTensor({1, 1, 3, 3}, DataType::Float32);
    Tensor w0 = gMkl->addTensor({1, 2, 3, 3}, DataType::Float32);
    auto conv = gMkl->addOp<ConvTransposed2dObj>(i0, w0, nullptr, 0, 0);

    gMkl->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());

    runtime->prepareAndRun(gMkl);
    EXPECT_TRUE(conv->getOutput()->equalData(vector<float>{
        0,   0,   1,   4,   4,   0,   6,   20,  26,  20,  9,   36,  84,
        84,  57,  36,  90,  164, 134, 80,  36,  84,  145, 112, 64,  0,
        9,   28,  31,  22,  27,  78,  155, 134, 83,  90,  225, 408, 327,
        192, 117, 270, 461, 350, 197, 90,  201, 334, 247, 136}));
}

TEST(mkl_ConvTransposed, tune) {
    Runtime runtime = MklRuntimeObj::getInstance();
    Graph gMkl = make_ref<GraphObj>(runtime);

    Tensor i0 = gMkl->addTensor({1, 448, 2, 2}, DataType::Float32);
    Tensor w0 = gMkl->addTensor({448, 256, 4, 4}, DataType::Float32);
    auto conv = gMkl->addOp<ConvTransposed2dObj>(i0, w0, nullptr);
    gMkl->dataMalloc();
    i0->setData(IncrementalGenerator());
    w0->setData(IncrementalGenerator());

    bool tune = true;
    runtime->prepareAndRun(gMkl, tune);
    // check record
    auto kernelAttrs =
        KernelAttrs{Device::MKL, conv->getOpType(), DataType::Float32};
    auto perfKey = PerfEngine::Key{kernelAttrs, conv->getOpPerfKey()};
    std::optional<PerfRecord> perfData =
        PerfEngine::getInstance().getPerfData(perfKey);
    ASSERT_TRUE(perfData.has_value());
}

} // namespace infini
