#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/tile.h"

#include "test.h"

namespace infini {

namespace {
/// Runs a Tile over `input` laid out as `dims` and returns what comes out.
vector<float> tiled(const Shape &dims, const vector<float> &input,
                    const Shape &repeats) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor in = g->addTensor(dims, DataType::Float32);
    auto op = g->addOp<TileObj>(in, nullptr, repeats);
    g->dataMalloc();
    in->copyin(input);
    runtime->run(g);
    return op->getOutput()->copyout<float>();
}
} // namespace

TEST(NativeCpuTile, RepeatsAlongTheLastAxis) {
    // Repeating the innermost axis lays the input's row down twice in a row.
    EXPECT_EQ(tiled({2, 2}, {1, 2, 3, 4}, {1, 2}),
              (vector<float>{1, 2, 1, 2, 3, 4, 3, 4}));
}

TEST(NativeCpuTile, RepeatsAlongTheFirstAxis) {
    // Repeating the outermost axis lays the whole input down twice.
    EXPECT_EQ(tiled({2, 2}, {1, 2, 3, 4}, {2, 1}),
              (vector<float>{1, 2, 3, 4, 1, 2, 3, 4}));
}

TEST(NativeCpuTile, RepeatsAlongBothAxes) {
    EXPECT_EQ(tiled({2, 2}, {1, 2, 3, 4}, {2, 2}),
              (vector<float>{1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4}));
}

TEST(NativeCpuTile, RepeatingOnceCopies) {
    EXPECT_EQ(tiled({2, 3}, {1, 2, 3, 4, 5, 6}, {1, 1}),
              (vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(NativeCpuTile, RepeatsAMiddleAxisOfThree) {
    // The axis in the middle is the one that tells a correct index mapping from
    // one that only happens to work at the ends.
    EXPECT_EQ(tiled({2, 1, 2}, {1, 2, 3, 4}, {1, 3, 1}),
              (vector<float>{1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}));
}

TEST(NativeCpuTile, RepeatingNoTimesLeavesNothing) {
    EXPECT_TRUE(tiled({2, 2}, {1, 2, 3, 4}, {0, 1}).empty());
}

TEST(NativeCpuTile, RepeatsAScalar) {
    EXPECT_EQ(tiled({1}, {7}, {4}), (vector<float>{7, 7, 7, 7}));
}

} // namespace infini
