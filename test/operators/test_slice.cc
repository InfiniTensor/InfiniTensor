#include "core/graph.h"
#include "core/runtime.h"
#include "operators/slice.h"
#include "test.h"
#include <limits>

namespace infini {
TEST(Slice, ShapeInference) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({10, 64, 162, 162}, DataType::UInt32);
        auto op = g->addOp<SliceObj>(i, nullptr, vector<int>{2, 9, 1, 5},
                                     vector<int>{3, 10, 100, 100}, std::nullopt,
                                     std::nullopt);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 1, 99, 95}));
    }
    {
        Graph g = make_ref<GraphObj>(cpuRuntime);
        Tensor i = g->addTensor({10, 64, 162, 162}, DataType::UInt32);
        auto op = g->addOp<SliceObj>(i, nullptr, vector<int>{2, 5},
                                     vector<int>{3, 100}, vector<int>{1, 3},
                                     std::nullopt);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{10, 1, 162, 95}));
    }
}

TEST(Slice, ComputedInt64BoundsAreClampedBeforeNarrowing) {
    Graph g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
    auto data = g->addTensor({5}, DataType::Int64);
    data->setShapeValue({10, 20, 30, 40, 50});
    auto starts = g->addTensor({1}, DataType::Int64);
    auto ends = g->addTensor({1}, DataType::Int64);
    starts->setShapeValue({std::numeric_limits<int64_t>::min()});
    ends->setShapeValue({std::numeric_limits<int64_t>::max()});
    auto op = g->addOp<SliceObj>(data, starts, ends, nullptr, vector<int>{0},
                                 std::nullopt);
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{5}));
    ASSERT_TRUE(op->getOutput()->getShapeValue().has_value());
    EXPECT_EQ(*op->getOutput()->getShapeValue(),
              (vector<int64_t>{10, 20, 30, 40, 50}));
}

TEST(Slice, ClampsForwardRangesAndEmptySlices) {
    for (const auto &bounds : vector<vector<int>>{
             {-99, -2, 0, 3, 3}, {4, 2, 4, 2, 0}, {9, 99, 5, 5, 0}}) {
        SCOPED_TRACE(bounds[0]);
        Graph g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
        auto data = g->addTensor({5}, DataType::Float32);
        auto op = g->addOp<SliceObj>(data, nullptr, vector<int>{bounds[0]},
                                     vector<int>{bounds[1]}, vector<int>{0},
                                     std::nullopt);
        EXPECT_EQ(op->getStarts(), (Shape{bounds[2]}));
        EXPECT_EQ(op->getEnds(), (Shape{bounds[3]}));
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{bounds[4]}));
    }
}

TEST(Slice, DynamicBoundsKeepSelectionDynamic) {
    Graph g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
    auto data = g->addTensor({4}, DataType::Int64);
    data->setShapeValue({10, 20, 30, 40});
    auto starts = g->addTensor({1}, DataType::Int64);
    auto ends = g->addTensor({1}, DataType::Int64);
    starts->setShapeValue({1}, {false});
    ends->setShapeValue({2}, {false});
    auto op = g->addOp<SliceObj>(data, starts, ends, nullptr, vector<int>{0},
                                 std::nullopt);
    for (int position : {1, 2}) {
        starts->setShapeValue({position}, {false});
        ends->setShapeValue({position + 1}, {false});
        g->shape_infer();
        ASSERT_TRUE(op->getOutput()->getShapeValue().has_value());
        EXPECT_EQ(*op->getOutput()->getShapeValue(),
                  (vector<int64_t>{10 * (position + 1)}));
        EXPECT_FALSE(op->getOutput()->isShapeValueWhollyFixed());
    }
}

TEST(Slice, RejectsInvalidOrRepeatedNormalizedAxes) {
    for (const auto &axes : vector<vector<int>>{{2}, {-3}, {0, -2}}) {
        Graph g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
        auto data = g->addTensor({3, 4}, DataType::Float32);
        EXPECT_THROW(
            g->addOp<SliceObj>(data, nullptr, vector<int>(axes.size(), 0),
                               vector<int>(axes.size(), 2), axes, std::nullopt),
            Exception);
    }
}

TEST(Slice, ReverseRangeShapeAvoidsStepOverflow) {
    Graph g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
    auto data = g->addTensor({5}, DataType::Int64);
    data->setShapeValue({10, 20, 30, 40, 50});
    for (int step : {-1, std::numeric_limits<int>::min()}) {
        auto op = g->addOp<SliceObj>(
            data, nullptr, vector<int>{std::numeric_limits<int>::max()},
            vector<int>{std::numeric_limits<int>::min()}, vector<int>{0},
            vector<int>{step});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{step == -1 ? 5 : 1}));
        ASSERT_TRUE(op->getOutput()->getShapeValue().has_value());
        EXPECT_EQ(*op->getOutput()->getShapeValue(),
                  step == -1 ? (vector<int64_t>{50, 40, 30, 20, 10})
                             : (vector<int64_t>{50}));
    }
}

} // namespace infini
