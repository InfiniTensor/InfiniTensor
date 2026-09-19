#include "core/graph.h"
#include "core/runtime.h"
#include "operators/slice.h"

#include "test.h"

namespace infini {

/// Slices the same input every way the kernel distinguishes between, and checks
/// each against the elements worked out by hand from the input's own layout.
/// The kernel copies runs of adjacent elements where it can, so which dimension
/// the run starts at is the thing worth varying: a whole trailing dimension, a
/// narrowed one, and a stepped one each take a different path through it.
void runSlice(const Shape &inputShape, const vector<int> &starts,
              const vector<int> &ends, const optional<vector<int>> &axes,
              const optional<vector<int>> &steps, const Shape &expectedShape,
              const vector<float> &expected) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(inputShape, DataType::Float32);
    auto op = g->addOp<SliceObj>(input, nullptr, starts, ends, axes, steps);
    g->dataMalloc();
    // Ascending whole numbers, so that every element says where it came from.
    input->setData(IncrementalGenerator());
    runtime->run(g);

    EXPECT_EQ(op->getOutput()->getDims(), expectedShape);
    EXPECT_TRUE(op->getOutput()->equalData(expected));
}

TEST(SliceNaive, WholeTrailingDimension) {
    // Rows 1 and 2 of four, taken whole: one adjacent run of ten.
    runSlice({4, 5}, {1}, {3}, vector<int>{0}, std::nullopt, {2, 5},
             {5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
}

TEST(SliceNaive, NarrowedTrailingDimension) {
    // A window within each row, so the runs are short and there are several.
    runSlice({3, 5}, {1}, {4}, vector<int>{1}, std::nullopt, {3, 3},
             {1, 2, 3, 6, 7, 8, 11, 12, 13});
}

TEST(SliceNaive, SteppedTrailingDimension) {
    // A step of two leaves nothing adjacent, so every element is placed on its
    // own and the run optimisation must not be taken.
    runSlice({2, 6}, {0}, {6}, vector<int>{1}, vector<int>{2}, {2, 3},
             {0, 2, 4, 6, 8, 10});
}

TEST(SliceNaive, SteppedOuterDimension) {
    // Every other row, each taken whole: adjacent runs at a stepped offset.
    runSlice({4, 3}, {0}, {4}, vector<int>{0}, vector<int>{2}, {2, 3},
             {0, 1, 2, 6, 7, 8});
}

TEST(SliceNaive, BothDimensionsNarrowed) {
    runSlice({4, 4}, {1, 2}, {3, 4}, vector<int>{0, 1}, std::nullopt, {2, 2},
             {6, 7, 10, 11});
}

TEST(SliceNaive, DropsTheTailOfAShape) {
    // What a model does to a shape vector: everything but the last dimension.
    // This is the case a simplifier writes, and it is one dimensional.
    runSlice({4}, {0}, {3}, vector<int>{0}, std::nullopt, {3}, {0, 1, 2});
}

TEST(SliceNaive, TakesTheWholeThing) {
    runSlice({2, 3}, {0, 0}, {2, 3}, vector<int>{0, 1}, std::nullopt, {2, 3},
             {0, 1, 2, 3, 4, 5});
}

TEST(SliceNaive, ThreeDimensionsWithASteppedMiddle) {
    // Input is 2x4x3, numbered 0..23. Taking rows 0 and 2 of the middle
    // dimension from both outer slices leaves four runs of three.
    runSlice({2, 4, 3}, {0}, {4}, vector<int>{1}, vector<int>{2}, {2, 2, 3},
             {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20});
}

TEST(SliceNaive, NegativeStartCountsFromTheEnd) {
    runSlice({5}, {-2}, {5}, vector<int>{0}, std::nullopt, {2}, {3, 4});
}

TEST(SliceNaive, EmptySliceDoesNoWork) {
    runSlice({5}, {2}, {2}, vector<int>{0}, std::nullopt, {0}, {});
}

TEST(SliceNaive, NegativeStepIsExplicitlyUnsupported) {
    EXPECT_THROW(runSlice({5}, {4}, {0}, vector<int>{0}, vector<int>{-1}, {4},
                          {4, 3, 2, 1}),
                 Exception);
}

} // namespace infini
