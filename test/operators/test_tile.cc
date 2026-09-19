#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/gather.h"
#include "operators/tile.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {

namespace {
/// A rank-1 tensor holding `values` as dimensions rather than data, wired in as
/// a weight so that shape inference can read it.
Tensor repeatsOf(const Graph &g, const vector<int64_t> &values) {
    Tensor t = g->addTensor({static_cast<int>(values.size())}, DataType::Int64);
    t->setShapeValue(values);
    return t;
}
} // namespace

TEST(Tile, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3}, DataType::Float32);
        auto op = g->addOp<TileObj>(i, nullptr, Shape{3, 2});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{6, 6}));
    }
    {
        // Repeating once leaves the dimension alone.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3, 4}, DataType::Float32);
        auto op = g->addOp<TileObj>(i, nullptr, Shape{1, 1, 1});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 4}));
    }
    {
        // Repeating no times empties that dimension.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3}, DataType::Float32);
        auto op = g->addOp<TileObj>(i, nullptr, Shape{0, 2});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{0, 6}));
    }
}

TEST(Tile, CountsMustDescribeTheInput) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor({2, 3}, DataType::Float32);
    // ONNX gives one count per input dimension, so a shorter list does not
    // describe this input at all and is refused rather than lined up.
    EXPECT_ANY_THROW(g->addOp<TileObj>(i, nullptr, Shape{2}));
    EXPECT_ANY_THROW(g->addOp<TileObj>(i, nullptr, Shape{2, -1}));
}

TEST(Tile, CountsFromAnEdgeAreReadEachTime) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i = g->addTensor({2, 3}, DataType::Float32);
    Tensor r = repeatsOf(g, {2, 4});
    auto op = g->addOp<TileObj>(i, r, nullptr);
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{4, 12}));
    EXPECT_EQ(op->getRepeats(), (Shape{2, 4}));

    // The counts are an edge of the graph, so a later shape inference reads
    // whatever they hold then rather than what they held at construction.
    r->setShapeValue({3, 1});
    g->shape_infer();
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{6, 3}));
}

TEST(Tile, WhichDimensionsFollowTheInput) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3}, DataType::Float32);
        auto op = g->addOp<TileObj>(i, nullptr, Shape{3, 1});
        // Each output dimension is a fixed multiple of the input's own, so it
        // varies exactly when that one does.
        auto zero = op->dimSources(0, 0);
        ASSERT_EQ(zero.size(), 1u);
        EXPECT_EQ(zero[0].input, 0u);
        EXPECT_EQ(zero[0].dim, 0u);
        auto one = op->dimSources(0, 1);
        ASSERT_EQ(one.size(), 1u);
        EXPECT_EQ(one[0].dim, 1u);
    }
    {
        // An emptied dimension is empty whatever the input held, so it follows
        // nothing and stays fixed however the input changes.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 3}, DataType::Float32);
        auto op = g->addOp<TileObj>(i, nullptr, Shape{0, 2});
        EXPECT_TRUE(op->dimSources(0, 0).empty());
        EXPECT_EQ(op->dimSources(0, 1).size(), 1u);
    }
}

TEST(Tile, CountsFromAnEdgeSettleOnlyTheirFixedElements) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    // A count worked out from the input's own shape, as an exporter writes it:
    // repeat the batch as many times as the batch is long, and the rest once.
    Tensor x = g->addTensor({4, 3}, DataType::Float32);
    x->setDimDescs(DimDescs{DimDesc{true, "batch"}, DimDesc{false, ""}});
    Tensor shape = g->addOp<ShapeObj>(x, nullptr)->getOutput();
    Tensor index = g->addTensor({1}, DataType::Int64);
    index->setShapeValue({0});
    Tensor head = g->addOp<GatherObj>(shape, index, nullptr, 0)->getOutput();
    Tensor tail = g->addTensor({1}, DataType::Int64);
    tail->setShapeValue({1});
    Tensor repeats =
        g->addOp<ConcatObj>(TensorVec{head, tail}, nullptr, 0)->getOutput();

    auto op = g->addOp<TileObj>(x, repeats, nullptr);
    g->shape_infer();
    EXPECT_EQ(op->getOutput()->getDims(), (Shape{16, 3}));

    // The first count follows the batch, so how many times that dimension is
    // repeated is not settled and neither is the dimension. The second is the
    // constant one, so the dimension it repeats stays as fixed as the input's.
    EXPECT_TRUE(op->getOutput()->isDimDynamic(0));
    EXPECT_FALSE(op->getOutput()->isDimDynamic(1));
}

} // namespace infini
