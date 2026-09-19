#include "core/graph.h"
#include "core/runtime.h"
#include "operators/resize.h"
#include "test.h"

namespace infini {
/// Resizing to a size the model asked for must reach that size whatever the
/// input turns out to be. A scale is a ratio against the input, so it describes
/// only the shape it was taken against: worked out once at construction, where
/// a dynamic dimension is still a placeholder, it would resize every later
/// shape by the placeholder's ratio instead of to the size requested.
TEST(Resize, SizesAreReachedWhateverTheInputWas) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    { // stretch: each axis reaches its own requested size.
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 1, 1, 1}, DataType::Float32);
        i->setDimDescs(
            {{false, ""}, {false, ""}, {true, "height"}, {true, "width"}});
        Tensor sizes = g->addTensor({4}, DataType::Int64);
        sizes->dataMalloc();
        sizes->copyin(vector<int64_t>{1, 1, 8, 8});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, std::nullopt, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::stretch);

        for (const auto &given : vector<Shape>{
                 {1, 1, 4, 4}, {1, 1, 2, 2}, {1, 1, 3, 7}, {1, 1, 8, 8}}) {
            i->setShape(given);
            g->shape_infer();
            EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 8, 8}))
                << "input " << vecToString(given);
        }
    }

    { // notLarger: one ratio across the resized axes, the smallest of them,
      // re-chosen for each input rather than kept from the first.
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 3, 1, 1}, DataType::Float32);
        i->setDimDescs(
            {{false, ""}, {false, ""}, {true, "height"}, {true, "width"}});
        Tensor sizes = g->addTensor({2}, DataType::Int64);
        sizes->dataMalloc();
        sizes->copyin(vector<int64_t>{7, 8});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, vector<int>{2, 3}, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::notLarger);

        for (const auto &[given, expected] :
             vector<std::pair<Shape, Shape>>{{{1, 3, 2, 4}, {1, 3, 4, 8}},
                                             {{1, 3, 4, 4}, {1, 3, 7, 7}},
                                             {{1, 3, 6, 3}, {1, 3, 7, 4}}}) {
            i->setShape(given);
            g->shape_infer();
            EXPECT_EQ(op->getOutput()->getDims(), expected)
                << "input " << vecToString(given);
        }
    }
}

/// An axis pinned to a requested size cannot be moved by any input, so it is
/// settled however the input dimension it replaces varies.
TEST(Resize, AnAxisResizedToASizeFollowsNothing) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(cpuRuntime);
    Tensor i = g->addTensor({1, 3, 4, 4}, DataType::Float32);
    i->setDimDescs(
        {{true, "batch"}, {false, ""}, {true, "height"}, {true, "width"}});
    Tensor sizes = g->addTensor({2}, DataType::Int64);
    sizes->dataMalloc();
    sizes->copyin(vector<int64_t>{8, 8});
    auto op = g->addOp<ResizeObj>(i, nullptr, vector<int>{2, 3}, sizes, nullptr,
                                  nullptr,
                                  ResizeObj::EKeepAspectRatioPolicy::stretch);
    g->shape_infer();

    const auto output = op->getOutput();
    EXPECT_TRUE(output->isDimDynamic(0));  // batch, passed through
    EXPECT_FALSE(output->isDimDynamic(1)); // channels, passed through
    EXPECT_FALSE(output->isDimDynamic(2)); // resized to 8 whatever came in
    EXPECT_FALSE(output->isDimDynamic(3));
}

/// Resizing by a scale is a multiple of the input, so it does follow it.
TEST(Resize, AnAxisResizedByAScaleFollowsTheInput) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(cpuRuntime);
    Tensor i = g->addTensor({1, 1, 2, 2}, DataType::Float32);
    i->setDimDescs(
        {{false, ""}, {false, ""}, {true, "height"}, {true, "width"}});
    Tensor scales = g->addTensor({4}, DataType::Float32);
    scales->dataMalloc();
    scales->copyin(vector<float>{1, 1, 2, 2});
    auto op =
        g->addOp<ResizeObj>(i, nullptr, std::nullopt, nullptr, scales, nullptr,
                            ResizeObj::EKeepAspectRatioPolicy::none);

    for (const auto &[given, expected] : vector<std::pair<Shape, Shape>>{
             {{1, 1, 4, 4}, {1, 1, 8, 8}}, {{1, 1, 3, 5}, {1, 1, 6, 10}}}) {
        i->setShape(given);
        g->shape_infer();
        EXPECT_EQ(op->getOutput()->getDims(), expected)
            << "input " << vecToString(given);
    }
    EXPECT_TRUE(op->getOutput()->isDimDynamic(2));
    EXPECT_TRUE(op->getOutput()->isDimDynamic(3));
}

TEST(Resize, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    // downsample_sizes_nearest no axes
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 1, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({4}, DataType::Int64);
        sizes->dataMalloc();
        sizes->copyin(vector<int64_t>{1, 1, 1, 3});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, std::nullopt, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::stretch);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 1, 3}));
    }
    // upsample_sizes_nearest with axes
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 1, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({2}, DataType::Int64);
        sizes->dataMalloc();
        sizes->copyin(vector<int64_t>{1, 3});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, vector<int>{2, 3}, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::stretch);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 1, 3}));
    }
    // upsample_sizes_nearest_notlarger
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 3, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({2}, DataType::Int64);
        sizes->dataMalloc();
        sizes->copyin(vector<int64_t>{7, 8});
        auto op = g->addOp<ResizeObj>(
            i, nullptr, vector<int>{2, 3}, sizes, nullptr, nullptr,
            ResizeObj::EKeepAspectRatioPolicy::notLarger);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 4, 8}));
    }
    // upsample_sizes_nearest_notsmaller
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({1, 3, 2, 4}, DataType::UInt32);
        Tensor sizes = g->addTensor({3}, DataType::Int64);
        sizes->dataMalloc();
        sizes->copyin(vector<int64_t>{2, 6, 8});
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
