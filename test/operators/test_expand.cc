#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/expand.h"
#include "operators/gather.h"
#include "operators/unary.h"

#include "test.h"

namespace infini {
namespace {
/// A rank-one integer constant holding a target, as an exporter writes one.
Tensor targetOf(const Graph &g, const vector<int64_t> &values) {
    auto t =
        g->addTensor(Shape{static_cast<int>(values.size())}, DataType::Int64);
    t->setShapeValue(values);
    t->dataMalloc();
    t->copyin(values);
    t->setWeight();
    return t;
}
} // namespace

/// A target read from an edge means the same thing as the same target given as
/// an attribute, and is read afresh whenever shapes are inferred.
TEST(Expand, ATargetFromAnEdgeIsReadEachTime) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({3, 1}, DataType::Float32);
        auto op = g->addOp<ExpandObj>(i, targetOf(g, {2, 1, 6}), nullptr);
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 6}));
    }
    { // An input dimension wider than the target's keeps its own size.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 1, 4}, DataType::Float32);
        i->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
        auto op = g->addOp<ExpandObj>(i, targetOf(g, {1, 3, 4}), nullptr);
        for (const auto &[given, expected] :
             vector<std::pair<Shape, Shape>>{{{1, 1, 4}, {1, 3, 4}},
                                             {{2, 1, 4}, {2, 3, 4}},
                                             {{7, 1, 4}, {7, 3, 4}}}) {
            i->setShape(given);
            g->shape_infer();
            EXPECT_EQ(op->getOutput()->getDims(), expected)
                << "input " << vecToString(given);
        }
    }
}

/// A dimension the target asks more than one of comes out at that number
/// whatever the input holds there, so it follows nothing a caller may vary. A
/// target of one leaves the input dimension alone, so that one is followed.
TEST(Expand, WhichDimensionsCanMoveFollowsTheTarget) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    { // target of one at the dynamic dimension: the output moves with it
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 1, 4}, DataType::Float32);
        i->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{1, 3, 4});
        g->shape_infer();
        const auto out = op->getOutput();
        EXPECT_TRUE(out->isDimDynamic(0));  // follows the batch
        EXPECT_FALSE(out->isDimDynamic(1)); // stretched to 3
        EXPECT_FALSE(out->isDimDynamic(2));
    }
    { // target above one at the dynamic dimension: broadcasting either agrees
      // with the target or stretches a one up to it, so the target settles it
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 1, 4}, DataType::Float32);
        i->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{2, 3, 4});
        g->shape_infer();
        const auto out = op->getOutput();
        EXPECT_FALSE(out->isDimDynamic(0));
        EXPECT_FALSE(out->isDimDynamic(1));
        EXPECT_FALSE(out->isDimDynamic(2));
    }
    { // a dynamic dimension in the middle, left alone by a target of one
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({2, 1, 4}, DataType::Float32);
        i->setDimDescs({{false, ""}, {true, "length"}, {false, ""}});
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{2, 1, 4});
        g->shape_infer();
        const auto out = op->getOutput();
        EXPECT_FALSE(out->isDimDynamic(0));
        EXPECT_TRUE(out->isDimDynamic(1));
        EXPECT_FALSE(out->isDimDynamic(2));
    }
    { // a target of a higher rank than the input: the dimensions it adds have
      // no input dimension to follow
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 4}, DataType::Float32);
        i->setDimDescs({{true, "length"}, {false, ""}});
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{5, 3, 4});
        g->shape_infer();
        const auto out = op->getOutput();
        EXPECT_FALSE(out->isDimDynamic(0)); // added by the target
        EXPECT_FALSE(out->isDimDynamic(1)); // stretched to 3
        EXPECT_FALSE(out->isDimDynamic(2));
    }
}

/// A target read from an edge says as much as a constant one where the element
/// it holds cannot change, and nothing where it can: which of the two settles
/// the dimension is not known until the number is.
TEST(Expand, ATargetFromAnEdgeSettlesOnlyItsFixedElements) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    { // A target that is wholly a constant, only reached through an edge. It
      // cannot move, so it settles exactly what the same attribute would.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 1, 4}, DataType::Float32);
        i->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
        auto op = g->addOp<ExpandObj>(i, targetOf(g, {1, 3, 4}), nullptr);
        g->shape_infer();
        const auto out = op->getOutput();
        EXPECT_TRUE(out->isDimDynamic(0));  // target of one, follows the batch
        EXPECT_FALSE(out->isDimDynamic(1)); // stretched to 3
        EXPECT_FALSE(out->isDimDynamic(2));
    }
    { // The batch element of the target is read off the input, so it moves with
      // it and cannot settle anything; the two constant elements still do.
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({1, 1, 4}, DataType::Float32);
        i->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
        auto shape = g->addOp<ShapeObj>(i, nullptr);
        auto batch = g->addOp<GatherObj>(shape->getOutput(), targetOf(g, {0}),
                                         nullptr, 0);
        auto target = g->addOp<ConcatObj>(
            TensorVec{batch->getOutput(), targetOf(g, {3, 4})}, nullptr, 0);
        auto op = g->addOp<ExpandObj>(i, target->getOutput(), nullptr);
        g->shape_infer();

        const auto value = target->getOutput();
        EXPECT_FALSE(value->isShapeValueFixed(0)); // the batch
        EXPECT_TRUE(value->isShapeValueFixed(1));
        EXPECT_TRUE(value->isShapeValueFixed(2));

        const auto out = op->getOutput();
        EXPECT_TRUE(out->isDimDynamic(0));
        EXPECT_FALSE(out->isDimDynamic(1));
        EXPECT_FALSE(out->isDimDynamic(2));

        // And the shape it works out is the one the input was given.
        for (const auto &[given, expected] : vector<std::pair<Shape, Shape>>{
                 {{2, 1, 4}, {2, 3, 4}}, {{6, 1, 4}, {6, 3, 4}}}) {
            i->setShape(given);
            g->shape_infer();
            EXPECT_EQ(out->getDims(), expected)
                << "input " << vecToString(given);
        }
    }
}

TEST(Expand, ShapeInference) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({3, 1}, DataType::Float32);
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{2, 1, 6});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 3, 6}));
    }
    {
        Graph g = make_ref<GraphObj>(runtime);
        Tensor i = g->addTensor({3, 1}, DataType::Float32);
        auto op = g->addOp<ExpandObj>(i, nullptr, Shape{3, 4});
        EXPECT_EQ(op->getOutput()->getDims(), (Shape{3, 4}));
    }
}

} // namespace infini
