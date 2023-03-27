#include "core/graph.h"
#include "core/runtime.h"
#include "operators/resize.h"
#include "test.h"

namespace infini {
TEST(Resize, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    // downsample_sizes_nearest no axes
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 1, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({4}, DataType::UInt32);
        sizes->dataMalloc();
        sizes->copyin(vector<uint32_t>{1, 1, 1, 3});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, std::nullopt, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::stretch);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 1, 3}));
    }
    // upsample_sizes_nearest with axes
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 1, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({2}, DataType::UInt32);
        sizes->dataMalloc();
        sizes->copyin(vector<uint32_t>{1, 3});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, vector<int>{2, 3}, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::stretch);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 1, 3}));
    }
    // upsample_sizes_nearest_notlarger
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 3, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({2}, DataType::UInt32);
        sizes->dataMalloc();
        sizes->copyin(vector<uint32_t>{7, 8});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, vector<int>{2, 3}, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::notLarger);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 4, 8}));
    }
    // upsample_sizes_nearest_notsmaller
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 3, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({3}, DataType::UInt32);
        sizes->dataMalloc();
        sizes->copyin(vector<uint32_t>{2, 6, 8});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, vector<int>{1, 2, 3}, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::notSmaller);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 9, 6, 12}));
    }
    // downsample_scales
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 1, 4, 4}, DataType::UInt32);
        Tensor scales = g->addTensor({3}, DataType::Float32);
        scales->dataMalloc();
        scales->copyin(vector<float>{1, 0.8, 0.8});
        auto op = g->addOp<ResizeObj>(i, nullptr, vector<int>{1, 2, 3}, nullptr,
                                      scales, nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 3, 3}));
    }
    // upsample_scales
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 1, 2, 2}, DataType::UInt32);
        Tensor scales = g->addTensor({4}, DataType::Float32);
        scales->dataMalloc();
        scales->copyin(vector<float>{1, 1, 2, 2});
        auto op = g->addOp<ResizeObj>(i, nullptr, std::nullopt, nullptr, scales,
                                      nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 4, 4}));
    }
}

} // namespace infini
