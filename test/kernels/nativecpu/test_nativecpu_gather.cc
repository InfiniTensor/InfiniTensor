#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/gather.h"

#include "test.h"

namespace infini {

/// Gathers from a 2x3x4 table of 0..23 and checks what comes back.
///
/// Picking along an axis has to carry the run of elements behind each picked
/// position along with it, so every axis is worth checking separately: an axis
/// in the middle is the one that has both a run before it and a run after it.
static void gatherIs(int axis, const vector<int64_t> &indices,
                     const Shape &indexShape, const vector<float> &expected) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto table = g->addTensor({2, 3, 4}, DataType::Float32);
    auto index = g->addTensor(indexShape, DataType::Int64);
    auto op = g->addOp<GatherObj>(table, index, nullptr, axis);
    g->dataMalloc();
    table->setData(IncrementalGenerator());
    index->copyin(indices);

    g->shape_infer();
    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(o->equalData(expected));
}

TEST(Gather, NativeCpuFirstAxis) {
    gatherIs(0, {1, 0}, Shape{2},
             {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
              0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11});
}

TEST(Gather, NativeCpuMiddleAxis) {
    gatherIs(1, {2, 0}, Shape{2},
             {8, 9, 10, 11, 0, 1, 2, 3, 20, 21, 22, 23, 12, 13, 14, 15});
}

TEST(Gather, NativeCpuLastAxis) {
    gatherIs(2, {3, 1}, Shape{2}, {3, 1, 7, 5, 11, 9, 15, 13, 19, 17, 23, 21});
}

TEST(Gather, NativeCpuNegativeIndexCountsBackFromTheEnd) {
    // -1 is the last position along the axis and -3 the first of three.
    gatherIs(1, {-1, -3}, Shape{2},
             {8, 9, 10, 11, 0, 1, 2, 3, 20, 21, 22, 23, 12, 13, 14, 15});
}

TEST(Gather, NativeCpuInt32Indices) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto table = g->addTensor({2, 3, 4}, DataType::Float32);
    auto index = g->addTensor({2}, DataType::Int32);
    auto op = g->addOp<GatherObj>(table, index, nullptr, 0);
    g->dataMalloc();
    table->setData(IncrementalGenerator());
    // ONNX allows either width of integer for the indices.
    index->copyin(vector<int32_t>{-1, -2});

    g->shape_infer();
    runtime->run(g);

    auto o = g->cloneTensor(op->getOutput(0));
    EXPECT_TRUE(o->equalData(vector<float>{12, 13, 14, 15, 16, 17, 18, 19,
                                           20, 21, 22, 23, 0,  1,  2,  3,
                                           4,  5,  6,  7,  8,  9,  10, 11}));
    index->copyin(vector<int32_t>{-3, 0});
    EXPECT_THROW(runtime->run(g), Exception);
}

} // namespace infini
