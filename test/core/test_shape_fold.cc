#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/pooling.h"
#include "operators/reshape.h"
#include "operators/slice.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include "test.h"

namespace infini {

namespace {
/// Data an initializer would arrive with, alongside the shape value the pass
/// reads. A constant the fold leaves in place is read by a kernel, which reads
/// data and not the shape value, so leaving the data out only ever looked right
/// where the pool happened to hold the wanted number already.
void fillFromShapeValue(const Tensor &t) {
    t->dataMalloc();
    t->copyin(*t->getShapeValue());
    t->setWeight();
}

/// A rank-one integer constant, as an initializer of a shape subgraph is.
Tensor constantOf(const Graph &g, const vector<int64_t> &values) {
    auto t =
        g->addTensor(Shape{static_cast<int>(values.size())}, DataType::Int64);
    t->setShapeValue(values);
    fillFromShapeValue(t);
    return t;
}

/// A single integer with no dimensions at all. An exporter indexes `Shape`
/// with one of these, so that the gather yields the axis as a number and the
/// `Unsqueeze` after it makes a one-element list.
Tensor scalarOf(const Graph &g, int64_t value) {
    auto t = g->addTensor(Shape{}, DataType::Int64);
    t->setShapeValue(vector<int64_t>{value});
    fillFromShapeValue(t);
    return t;
}
} // namespace

/// The mask says which elements of a shape value cannot change. A dimension
/// the model declared fixed cannot, because `validateShapeChange` rejects any
/// attempt to; one declared dynamic can.
TEST(ShapeFold, MaskFollowsDeclaredDimensions) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 224, 224}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}, {true, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);

    const auto &value = shape->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{1, 3, 224, 224}));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(2));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(3));
    EXPECT_FALSE(shape->getOutput()->isShapeValueWhollyFixed());
}

/// A `Shape` reading something the graph worked out is settled when everything
/// that was worked out from is. An exporter puts the shape chain after the
/// layers whose output it describes -- a picture model reads the size a
/// convolution left, not the size it was given -- so without this the chain
/// that matters most is the one nothing can be known about.
///
/// This is settled by a pass over the graph rather than as each operator is
/// added, because a tensor's producers have to exist before what it inherits
/// from them can be read. So it takes a round of inference.
TEST(ShapeFold, FixednessReachesComputedTensors) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}, {false, ""}, {false, ""}});
    auto computed = g->addOp<ReluObj>(input, nullptr);
    auto shape = g->addOp<ShapeObj>(computed->getOutput(), nullptr);
    g->shape_infer();

    EXPECT_TRUE(shape->getOutput()->isShapeValueWhollyFixed());
    EXPECT_EQ(*shape->getOutput()->getShapeValue(),
              (vector<int64_t>{1, 3, 8, 8}));
}

/// One dimension that moves unsettles that dimension and no other. Which
/// output dimension an input dimension reached is a question only the operator
/// can answer, and `dimSources` is where it answers it: a `Relu` is applied in
/// place, so a dimension of what it computes is the dimension it was.
///
/// This matters because a batch size that moves is the ordinary case. Under a
/// rule that unsettles the whole tensor, a chain reading a computed tensor only
/// settles once every dimension it descends from is pinned -- so a deployment
/// serving one picture size at a batch size it varies would settle nothing at
/// all, though its height and width plainly cannot move.
///
/// An operator that says nothing still gets the conservative answer, which is
/// every dimension of every input; being wrong that way costs a fold, while the
/// other way round would fold something that still has to be computed.
TEST(ShapeFold, OneDimensionThatMovesUnsettlesOnlyItself) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3}, DataType::Float32);
    // Only the batch moves. The three is a number the model was given.
    input->setDimDescs({{true, "batch"}, {false, ""}});
    auto computed = g->addOp<ReluObj>(input, nullptr);
    auto shape = g->addOp<ShapeObj>(computed->getOutput(), nullptr);
    g->shape_infer();

    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    // Fixed read straight off the input, and fixed read through the Relu too.
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_FALSE(shape->getOutput()->isShapeValueWhollyFixed());

    auto direct = g->addOp<ShapeObj>(input, nullptr);
    EXPECT_TRUE(direct->getOutput()->isShapeValueFixed(1));
}

/// An operator that has not said which dimensions it works out from keeps the
/// conservative answer, so adding `dimSources` to some operators cannot settle
/// a dimension downstream of one that stayed quiet.
TEST(ShapeFold, AnOperatorThatSaysNothingUnsettlesEverything) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 3}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {false, ""}});
    // Transpose does not override `dimSources`, though its answer would be the
    // permutation. Until it does, both dimensions of what it computes are read
    // as replaceable -- including the one that came from the fixed three.
    auto moved = g->addOp<TransposeObj>(input, nullptr, vector<int>{1, 0});
    auto shape = g->addOp<ShapeObj>(moved->getOutput(), nullptr);
    g->shape_infer();

    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(1));
}

/// The case per-dimension answers were added for: a deployment that serves one
/// picture size at a batch size it varies. Height and width are pinned, batch
/// is not, and what the fold settles has to survive the batch moving afterwards
/// -- settling something the batch reaches would be a wrong answer rather than
/// a slow one.
TEST(ShapeFold, PinningSomeDimensionsSettlesOnlyWhatFollowsThem) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    // Height and width pinned, batch still free. This is what `pin_dims` leaves
    // behind once a deployment has said which size it serves.
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto weight = g->addTensor(Shape{4, 3, 3, 3}, DataType::Float32);
    weight->dataMalloc();
    weight->setWeight();
    auto conv = g->addOp<ConvObj>(input, weight, nullptr, 1, 1);
    auto pooled = g->addOp<MaxPoolObj>(conv->getOutput(), nullptr, 2, 2, 1, 1,
                                       0, 0, 2, 2, 0);
    auto activated = g->addOp<ReluObj>(pooled->getOutput(), nullptr);
    auto shape = g->addOp<ShapeObj>(activated->getOutput(), nullptr);
    g->shape_infer();

    // Batch reaches dimension zero and nothing else. Channels come from the
    // weight, and the spatial dimensions from sizes that were pinned.
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(2));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(3));

    // And it stays that way once the batch actually moves: the settled
    // dimensions hold their values while dimension zero follows.
    const auto settled = *shape->getOutput()->getShapeValue();
    input->setShape(Shape{7, 3, 8, 8});
    g->shape_infer();
    const auto after = *shape->getOutput()->getShapeValue();
    EXPECT_EQ(after[0], 7);
    EXPECT_EQ(after[1], settled[1]);
    EXPECT_EQ(after[2], settled[2]);
    EXPECT_EQ(after[3], settled[3]);
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(3));
}

/// A global pool covers whatever it is given in one window, so its spatial
/// output is a single position however large or dynamic the input was. That is
/// settled without being told anything about the input, which is what makes it
/// worth saying: the classifier at the end of an image model sits behind one,
/// and a shape read after it is the same shape at every picture size.
TEST(ShapeFold, AGlobalPoolSettlesItsSpatialOutput) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    // Nothing pinned: every dimension of the picture may move.
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {true, "H"}, {true, "W"}});
    // A global pool is an average pool told its window is the whole input.
    auto pooled =
        g->addOp<AvgPoolObj>(input, nullptr, 8, 8, 1, 1, 0, 0, 1, 1, 0, true);
    auto shape = g->addOp<ShapeObj>(pooled->getOutput(), nullptr);
    g->shape_infer();

    EXPECT_EQ(pooled->getOutput()->getDims(), (Shape{1, 3, 1, 1}));
    // Batch still follows the input and channels are passed through, but the
    // two spatial dimensions are one whatever arrives.
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(2));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(3));

    // The window follows the input rather than being what construction was
    // passed, so a spatial size it never saw still pools to one position. A
    // window frozen at construction gave back the input unchanged here.
    input->setShape(Shape{1, 3, 16, 16});
    g->shape_infer();
    EXPECT_EQ(pooled->getOutput()->getDims(), (Shape{1, 3, 1, 1}));
    EXPECT_EQ(pooled->getKh(), 16);
    EXPECT_EQ(pooled->getKw(), 16);
    const auto after = *shape->getOutput()->getShapeValue();
    EXPECT_EQ(after[2], 1);
    EXPECT_EQ(after[3], 1);

    // An ordinary pool keeps the window it was given and strides over whatever
    // it is handed, so its spatial output follows the input and is not settled.
    Graph g2 = make_ref<GraphObj>(runtime);
    auto input2 = g2->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input2->setDimDescs(
        {{true, "batch"}, {false, ""}, {true, "H"}, {true, "W"}});
    auto plain =
        g2->addOp<AvgPoolObj>(input2, nullptr, 2, 2, 1, 1, 0, 0, 2, 2, 0);
    auto shape2 = g2->addOp<ShapeObj>(plain->getOutput(), nullptr);
    g2->shape_infer();
    EXPECT_TRUE(shape2->getOutput()->isShapeValueFixed(1));
    EXPECT_FALSE(shape2->getOutput()->isShapeValueFixed(2));
    EXPECT_FALSE(shape2->getOutput()->isShapeValueFixed(3));
}

/// A weight is the data it carries and its shape is that data's shape. Nothing
/// hands one a shape -- only an input is ever given one -- so its dimensions
/// are fixed whatever it was or was not told about them. Otherwise fixedness
/// would stop at the first layer that has weights, which is most of them.
TEST(ShapeFold, FixednessCrossesALayerWithWeights) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 3}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}});
    auto weight = g->addTensor(Shape{3, 4}, DataType::Float32);
    weight->dataMalloc();
    weight->setWeight();
    EXPECT_FALSE(weight->isDimDynamic(0));

    auto product =
        g->addOp<MatmulObj>(input, weight, nullptr, false, false, nullptr);
    auto shape = g->addOp<ShapeObj>(product->getOutput(), nullptr);
    g->shape_infer();

    EXPECT_TRUE(shape->getOutput()->isShapeValueWhollyFixed());
    EXPECT_EQ(*shape->getOutput()->getShapeValue(), (vector<int64_t>{2, 4}));
}

/// A tensor that never declared its dimensions keeps every one of them
/// replaceable, so nothing about it is settled and nothing may be folded.
TEST(ShapeFold, UndeclaredDimensionsAreNotFixed) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3}, DataType::Float32);
    auto shape = g->addOp<ShapeObj>(input, nullptr);

    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
}

/// Gathering a settled dimension gives a settled element; gathering one that
/// moves does not. This is the case an exporter produces: it reads every axis
/// off `Shape`, including the ones the model fixed.
TEST(ShapeFold, GatherIsAsFixedAsWhatItPicked) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {true, "height"}, {true, "width"}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto channels =
        g->addOp<GatherObj>(shape->getOutput(), constantOf(g, {1}), nullptr, 0);
    auto batch =
        g->addOp<GatherObj>(shape->getOutput(), constantOf(g, {0}), nullptr, 0);

    EXPECT_TRUE(channels->getOutput()->isShapeValueWhollyFixed());
    EXPECT_FALSE(batch->getOutput()->isShapeValueWhollyFixed());

    // Only the settled branch goes. The one reading the batch size has to be
    // there to read it again when the batch size changes.
    const auto before = g->getOperators().size();
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 1u);
    EXPECT_EQ(g->getOperators().size(), before - 1);
    EXPECT_TRUE(g->checkValid());
}

/// A whole settled chain folds, not merely its first step, because the mask
/// carries along it during shape inference.
TEST(ShapeFold, SettledChainFoldsThroughout) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto height =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto lifted =
        g->addOp<UnsqueezeObj>(height->getOutput(), nullptr, Shape{0});
    auto joined = g->addOp<ConcatObj>(
        TensorVec{lifted->getOutput(), constantOf(g, {4})}, nullptr, 0);

    EXPECT_TRUE(joined->getOutput()->isShapeValueWhollyFixed());
    EXPECT_EQ(*joined->getOutput()->getShapeValue(), (vector<int64_t>{8, 4}));

    EXPECT_EQ(g->getOperators().size(), 4u);
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 4u);
    EXPECT_EQ(g->getOperators().size(), 0u);
    EXPECT_TRUE(g->checkValid());

    // The result is still there to be read, and now stands on its own.
    EXPECT_EQ(*joined->getOutput()->getShapeValue(), (vector<int64_t>{8, 4}));
    EXPECT_EQ(joined->getOutput()->getSource(), nullptr);
}

/// A `Concat` joining a settled dimension with one that moves keeps running,
/// and its kernel reads its inputs, so the folded one must hold real bytes.
TEST(ShapeFold, FoldedTensorCarriesItsDataForALiveConsumer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto batch =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 0), nullptr, 0);
    auto height =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto liveBranch =
        g->addOp<UnsqueezeObj>(batch->getOutput(), nullptr, Shape{0});
    auto settledBranch =
        g->addOp<UnsqueezeObj>(height->getOutput(), nullptr, Shape{0});
    auto joined = g->addOp<ConcatObj>(
        TensorVec{liveBranch->getOutput(), settledBranch->getOutput()}, nullptr,
        0);

    // The join reads the batch size, so it is not settled and stays.
    EXPECT_FALSE(joined->getOutput()->isShapeValueWhollyFixed());
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 2u);
    EXPECT_TRUE(g->checkValid());

    g->dataMalloc();
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              (vector<int64_t>{1, 8}));
}

/// A folded constant has to outlive every allocation that follows, because
/// nothing runs to fill it again. Allocation plans storage for a tensor with no
/// producer and hands the offset back once the last reader has run, which is
/// right for something refilled from outside each run and wrong for this.
///
/// Whether the offset is then reused at all decides whether the fault shows, so
/// the graph is built to leave no choice: folding the settled branch leaves one
/// constant of eight elements, and the operator planned after the join asks for
/// exactly the sixty-four bytes the join has just released. Every other hole
/// here is smaller, so any allocator must take that one.
TEST(ShapeFold, FoldedConstantKeepsItsStorage) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 16}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto batch =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 0), nullptr, 0);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 1), nullptr, 0);

    // Eight copies of the settled width, so that what folding leaves is a
    // constant large enough to be the only hole worth reusing.
    TensorVec settledParts;
    for (int i = 0; i < 8; ++i) {
        settledParts.push_back(
            g->addOp<UnsqueezeObj>(width->getOutput(), nullptr, Shape{0})
                ->getOutput());
    }
    auto settledList = g->addOp<ConcatObj>(settledParts, nullptr, 0);
    auto liveOne =
        g->addOp<UnsqueezeObj>(batch->getOutput(), nullptr, Shape{0});
    auto joined = g->addOp<ConcatObj>(
        TensorVec{liveOne->getOutput(), settledList->getOutput()}, nullptr, 0);

    // Added last so that it is planned after the join has released the
    // constant, and shaped so that it asks for just as much.
    auto big = g->addOp<ReluObj>(input, nullptr);
    ASSERT_EQ(big->getOutput()->getBytes(),
              settledList->getOutput()->getBytes());

    // The settled gather, its eight unsqueezes and their join.
    ASSERT_EQ(g->foldFixedShapeSubgraph(), 10u);
    ASSERT_TRUE(g->checkValid());

    const vector<int64_t> expected{1, 16, 16, 16, 16, 16, 16, 16, 16};
    g->dataMalloc();
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              expected);

    // The run above is what hands the constant's storage to another tensor, so
    // it is the second run that reads what became of it.
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              expected);

    // A new batch size replans every offset, which the folded width must also
    // survive, twice over for the same reason.
    input->setShape({4, 16});
    g->shape_infer();
    g->dataMalloc();
    const vector<int64_t> reshaped{4, 16, 16, 16, 16, 16, 16, 16, 16};
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              reshaped);
    runtime->run(g);
    EXPECT_EQ(g->cloneTensor(joined->getOutput())->copyout<int64_t>(),
              reshaped);
}

/// A settled result asked for as an output of the graph keeps what produces it.
/// The numbers in it would be right either way, but the graph promises that
/// each tensor it holds is either produced by an operator or given from
/// outside, and an output with neither would break that promise -- as
/// `checkValid` here insists.
TEST(ShapeFold, GraphOutputKeepsItsProducer) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 3}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    shape->getOutput()->setOutput();

    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_EQ(g->getOperators().size(), 1u);
    EXPECT_TRUE(g->checkValid());
    const auto &tensors = g->getTensors();
    EXPECT_NE(std::find(tensors.begin(), tensors.end(), shape->getOutput()),
              tensors.end());
    EXPECT_EQ(*shape->getOutput()->getShapeValue(), (vector<int64_t>{2, 3}));
}

/// Folding must not answer differently the second time, so that running the
/// pass again on an already folded graph is safe.
TEST(ShapeFold, FoldingTwiceChangesNothingFurther) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 3}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}});
    g->addOp<ShapeObj>(input, nullptr);

    EXPECT_EQ(g->foldFixedShapeSubgraph(), 1u);
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_TRUE(g->checkValid());
}

/// A shape computation works with the dimensions it read, not only with the
/// dimensions themselves: an exporter writes `d // heads` as a division on the
/// number `Shape` produced. The answer is as settled as both of the numbers it
/// came from, so a fixed dimension over a constant is settled.
TEST(ShapeFold, ArithmeticOnDimensionsCarriesItsValue) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto perHead =
        g->addOp<DivObj>(width->getOutput(), scalarOf(g, 4), nullptr);

    const auto &value = perHead->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{8}));
    EXPECT_TRUE(perHead->getOutput()->isShapeValueWhollyFixed());
}

/// The same division reading a dimension that moves. Its answer moves with it,
/// so nothing about it is settled and it must keep running.
TEST(ShapeFold, ArithmeticIsUnsettledWhenEitherSideIs) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto seq =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 1), nullptr, 0);
    auto doubled = g->addOp<MulObj>(seq->getOutput(), scalarOf(g, 2), nullptr);

    const auto &value = doubled->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{14}));
    // The number is known for the shape the graph currently has, and is not
    // the same under every shape it may be given.
    EXPECT_FALSE(doubled->getOutput()->isShapeValueFixed(0));
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_TRUE(g->checkValid());
}

/// A division with nothing to divide by. There is no number to record, so none
/// is recorded and the operator is left to run as it always did.
TEST(ShapeFold, DivisionByZeroYieldsNoValue) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto broken = g->addOp<DivObj>(width->getOutput(), scalarOf(g, 0), nullptr);

    EXPECT_FALSE(broken->getOutput()->getShapeValue().has_value());
    EXPECT_FALSE(broken->getOutput()->isShapeValueWhollyFixed());
    EXPECT_TRUE(g->checkValid());
}

/// A copy hands on both the numbers and how settled they are. An exporter puts
/// one in the middle of a shape subgraph whenever it casts a value to the type
/// it already has, and a chain broken there would leave the `Reshape` below it
/// without a target.
TEST(ShapeFold, CopyPassesShapeValueThrough) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 224, 224}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto height =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto copied = g->addOp<IdentityObj>(height->getOutput(), nullptr);

    const auto &value = copied->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{224}));
    EXPECT_TRUE(copied->getOutput()->isShapeValueWhollyFixed());
}

/// The fold takes the arithmetic with it. `32 / 4` is the same eight under
/// every shape the graph may be given, so nothing needs to work it out twice.
TEST(ShapeFold, SettledArithmeticFolds) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 7, 32}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto width =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto perHead =
        g->addOp<DivObj>(width->getOutput(), scalarOf(g, 4), nullptr);
    auto lifted =
        g->addOp<UnsqueezeObj>(perHead->getOutput(), nullptr, vector<int>{0});
    lifted->getOutput()->setOutput();

    // The gather, the division, and the `Shape` left with nobody to read it.
    // The unsqueeze is what the graph was asked for, so it keeps its place and
    // reads the eight the division left behind.
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 3u);
    EXPECT_EQ(g->getOperators().size(), 1u);
    EXPECT_TRUE(g->checkValid());
    EXPECT_EQ(perHead->getOutput()->copyout<int64_t>(), (vector<int64_t>{8}));
}

/// Taking part of a shape carries the elements that were taken, each as settled
/// as it was. This is what a model writes for "everything but the last
/// dimension", and a simplifier produces it even where the model did not.
TEST(ShapeFold, SliceCarriesThePartItTook) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 16, 8}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    // The leading two dimensions: one that moves and one that does not.
    auto head =
        g->addOp<SliceObj>(shape->getOutput(), nullptr, vector<int>{0},
                           vector<int>{2}, vector<int>{0}, std::nullopt);

    const auto &value = head->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{1, 16}));
    EXPECT_FALSE(head->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(head->getOutput()->isShapeValueFixed(1));
    EXPECT_FALSE(head->getOutput()->isShapeValueWhollyFixed());
}

/// A slice of settled elements alone is settled, and so folds away with the
/// chain that produced it.
TEST(ShapeFold, SettledSliceFolds) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 16, 8}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    // The trailing two, both declared fixed.
    auto tail =
        g->addOp<SliceObj>(shape->getOutput(), nullptr, vector<int>{1},
                           vector<int>{3}, vector<int>{0}, std::nullopt);
    EXPECT_TRUE(tail->getOutput()->isShapeValueWhollyFixed());
    EXPECT_EQ(*tail->getOutput()->getShapeValue(), (vector<int64_t>{16, 8}));

    auto live = g->addOp<ReluObj>(input, nullptr);
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 2u);
    EXPECT_EQ(live->getOutput()->getDims(), (Shape{1, 16, 8}));
}

/// A step walks the range, so the elements taken are the ones it lands on and
/// the fixedness of each follows the element it came from.
TEST(ShapeFold, SteppedSliceTakesWhatItLandsOn) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 4, 6, 8}, DataType::Float32);
    input->setDimDescs(
        {{false, ""}, {true, "height"}, {false, ""}, {true, "width"}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto every2 =
        g->addOp<SliceObj>(shape->getOutput(), nullptr, vector<int>{0},
                           vector<int>{4}, vector<int>{0}, vector<int>{2});

    const auto &value = every2->getOutput()->getShapeValue();
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, (vector<int64_t>{2, 6}));
    EXPECT_TRUE(every2->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(every2->getOutput()->isShapeValueFixed(1));
}

/// Slicing anything of higher rank is a slice of data rather than of
/// dimensions, and nothing here can say what the result would hold. Such a
/// tensor cannot carry a shape value to begin with -- only a rank one integer
/// can -- so the result carries none either, and the fold leaves it alone.
TEST(ShapeFold, SliceOfDataYieldsNoValue) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto table = g->addTensor(Shape{4, 4}, DataType::Int64);
    EXPECT_FALSE(table->canHoldShapeValue());
    auto part =
        g->addOp<SliceObj>(table, nullptr, vector<int>{0}, vector<int>{2},
                           vector<int>{0}, std::nullopt);
    EXPECT_FALSE(part->getOutput()->getShapeValue().has_value());
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
}

/// A `Reshape` may read its target through a slice, which is the shape the
/// chain a simplifier leaves behind has: the shape of a tensor, its tail
/// dropped, joined with a constant.
TEST(ShapeFold, ReshapeReadsATargetThroughASlice) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{2, 3, 4}, DataType::Float32);
    input->setDimDescs({{true, "batch"}, {true, "seq"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto head =
        g->addOp<SliceObj>(shape->getOutput(), nullptr, vector<int>{0},
                           vector<int>{2}, vector<int>{0}, std::nullopt);
    auto target = g->addOp<ConcatObj>(
        TensorVec{head->getOutput(), constantOf(g, {4})}, nullptr, 0);
    auto reshaped = g->addOp<ReshapeObj>(input, target->getOutput(), nullptr);
    EXPECT_EQ(reshaped->getOutput()->getDims(), (Shape{2, 3, 4}));

    // And it follows the input, which is the whole point of reading the target
    // at run time rather than settling it when the graph was built.
    input->setShape({5, 7, 4});
    g->shape_infer();
    EXPECT_EQ(reshaped->getOutput()->getDims(), (Shape{5, 7, 4}));
}

/// The operators a shape computation is built from are the operators real data
/// uses too: an attention export squeezes and scales activations with the same
/// `Squeeze` and `Mul` an exporter joins dimensions with. Counting by type
/// alone reported those as part of the shape subgraph, which a fold can never
/// reach, so it claimed a subgraph larger than the one that exists -- and the
/// number it left behind never fell to zero however completely the real one
/// folded.
TEST(ShapeFold, TheCountLeavesOutArithmeticOnData) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // A shape computation: settled, and countable.
    auto input = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    auto height =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto lifted =
        g->addOp<UnsqueezeObj>(height->getOutput(), nullptr, Shape{0});

    // And a `Mul` on the activations themselves, of a type the count includes,
    // carrying no shape value because a rank-four float tensor cannot hold one.
    auto scale = g->addTensor(Shape{1, 3, 8, 8}, DataType::Float32);
    auto scaled = g->addOp<MulObj>(input, scale, nullptr);
    EXPECT_FALSE(scaled->getOutput()->getShapeValue().has_value());

    EXPECT_EQ(g->getOperators().size(), 4u);
    EXPECT_EQ(g->shapeSubgraphSize(), 3u);

    // So a fold that reaches everything it can leaves nothing counted, rather
    // than leaving the arithmetic behind as though a shape were still in it.
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 3u);
    EXPECT_EQ(g->shapeSubgraphSize(), 0u);
    EXPECT_EQ(g->getOperators().size(), 1u);
    EXPECT_TRUE(g->checkValid());

    // The value the chain worked out is still readable where it ended.
    EXPECT_EQ(*lifted->getOutput()->getShapeValue(), (vector<int64_t>{8}));
}

/// Keeping a mixed shape whole avoids adding work to reconstruct values that
/// the original Shape operator still has to compute.
TEST(ShapeFold, PartiallyFixedShapeStaysWhole) {
    Graph g = make_ref<GraphObj>(NativeCpuRuntimeObj::getInstance());
    auto input = g->addTensor({2, 3, 4, 5}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr)->getOutput();
    auto flat = g->addOp<FlattenObj>(input, nullptr, 0)->getOutput();
    auto reshaped = g->addOp<ReshapeObj>(flat, shape, nullptr);
    g->shape_infer();

    EXPECT_FALSE(shape->isShapeValueFixed(0));
    EXPECT_TRUE(shape->isShapeValueFixed(1));
    EXPECT_TRUE(shape->isShapeValueFixed(2));
    EXPECT_TRUE(shape->isShapeValueFixed(3));
    const auto before = g->getOperators().size();
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_EQ(g->getOperators().size(), before);
    EXPECT_EQ(reshaped->getInputs(1), shape);
    EXPECT_EQ(g->shapeSubgraphSize(), 1u);
    input->setShape({5, 3, 4, 5});
    g->shape_infer();
    EXPECT_EQ(reshaped->getOutput()->getDims(), (Shape{5, 3, 4, 5}));
    EXPECT_EQ(*shape->getShapeValue(), (vector<int64_t>{5, 3, 4, 5}));
    EXPECT_TRUE(g->checkValid());
}

/// Interleaved fixed dimensions must not freeze the dimensions between them.
TEST(ShapeFold, InterleavedShapePreservesDynamicDimensions) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 64, 56, 56}, DataType::Float32);
    input->setDimDescs(
        {{true, "batch"}, {false, ""}, {true, "H"}, {true, "W"}});

    auto shape = g->addOp<ShapeObj>(input, nullptr);
    g->shape_infer();

    // Element 0 (batch) is dynamic, element 1 (channels) is fixed,
    // elements 2-3 (H/W) are dynamic
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(0));
    EXPECT_TRUE(shape->getOutput()->isShapeValueFixed(1));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(2));
    EXPECT_FALSE(shape->getOutput()->isShapeValueFixed(3));

    // Use Flatten to create data that automatically adjusts with input
    auto flattened = g->addOp<FlattenObj>(input, nullptr, 0);
    g->shape_infer();

    auto reshaped = g->addOp<ReshapeObj>(flattened->getOutput(),
                                         shape->getOutput(), nullptr);
    g->shape_infer();

    // No complete result is fixed, so the shape computation stays live.
    g->foldFixedShapeSubgraph();

    // Test multiple shape updates
    const vector<Shape> testShapes = {
        {2, 64, 56, 56},
        {4, 64, 112, 112},
        {1, 64, 28, 28},
        {8, 64, 224, 224},
    };

    for (const auto &testShape : testShapes) {
        input->setShape(testShape);
        g->shape_infer();
        EXPECT_EQ(reshaped->getOutput()->getDims(), testShape);

        const auto shapeVal = *reshaped->getInputs(1)->getShapeValue();
        EXPECT_EQ(shapeVal[0], testShape[0]);
        EXPECT_EQ(shapeVal[1], testShape[1]);
        EXPECT_EQ(shapeVal[2], testShape[2]);
        EXPECT_EQ(shapeVal[3], testShape[3]);
    }
}

/// A fixed selection folds even when another consumer needs the mixed shape.
TEST(ShapeFold, GatherOnPartiallyFixedShape) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor({3, 4, 5, 2}, DataType::Float32);
    input->setInput();
    input->setDimDescs(
        {{false, ""}, {false, ""}, {false, ""}, {true, "batch"}});
    auto shape = g->addOp<ShapeObj>(input, nullptr)->getOutput();
    auto flat = g->addOp<FlattenObj>(input, nullptr, 0)->getOutput();
    auto reshaped = g->addOp<ReshapeObj>(flat, shape, nullptr)->getOutput();
    reshaped->setOutput();
    auto fixed =
        g->addOp<GatherObj>(shape, scalarOf(g, 1), nullptr, 0)->getOutput();
    auto observed = g->addOp<IdentityObj>(fixed, nullptr)->getOutput();
    observed->setOutput();
    g->shape_infer();

    const auto before = g->getOperators().size();
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 1u);
    EXPECT_EQ(g->getOperators().size(), before - 1);
    EXPECT_FALSE(shape->isShapeValueWhollyFixed());
    EXPECT_TRUE(fixed->isWeight());
    for (int batch : {2, 5, 1}) {
        input->setShape({3, 4, 5, batch});
        g->shape_infer();
        g->dataMalloc();
        input->setData(IncrementalGenerator());
        runtime->run(g);
        EXPECT_EQ(observed->copyout<int64_t>(), (vector<int64_t>{4}));
        EXPECT_EQ(reshaped->getDims(), (Shape{3, 4, 5, batch}));
        EXPECT_TRUE(reshaped->equalData(input));
    }
    EXPECT_TRUE(g->checkValid());
}

/// One fixed dimension is enough to fold its consumer, without splitting the
/// Shape tensor needed by a dynamic consumer.
TEST(ShapeFold, SingleFixedElementFolding) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor({2, 10, 32}, DataType::Float32);
    input->setInput();
    input->setDimDescs({{true, "batch"}, {true, "length"}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr)->getOutput();
    auto fixed =
        g->addOp<GatherObj>(shape, scalarOf(g, -1), nullptr, 0)->getOutput();
    auto observed = g->addOp<IdentityObj>(fixed, nullptr)->getOutput();
    observed->setOutput();
    auto batch =
        g->addOp<GatherObj>(shape, scalarOf(g, 0), nullptr, 0)->getOutput();
    batch->setOutput();
    g->shape_infer();
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 1u);

    for (const auto &dims :
         vector<Shape>{{4, 20, 32}, {1, 5, 32}, {8, 100, 32}}) {
        input->setShape(dims);
        g->shape_infer();
        g->dataMalloc();
        runtime->run(g);
        EXPECT_EQ(observed->copyout<int64_t>(), (vector<int64_t>{32}));
        EXPECT_EQ(batch->copyout<int64_t>(), (vector<int64_t>{dims[0]}));
    }
    EXPECT_TRUE(g->checkValid());
}

/// Gather(Constant, const_index) should be replaced with a Constant directly
TEST(ShapeFold, GatherOnConstantBecomesConstant) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    // Input with fully fixed shape
    auto input = g->addTensor(Shape{3, 224, 224, 32}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}, {false, ""}, {false, ""}});

    // Shape operator - output will be fully fixed [3, 224, 224, 32]
    auto shape = g->addOp<ShapeObj>(input, nullptr);
    g->shape_infer();

    // Gather extracting index 2 (value 224)
    auto gathered =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    g->shape_infer();

    // Add a consumer to keep track of the result
    auto consumer = g->addOp<IdentityObj>(gathered->getOutput(), nullptr);
    g->shape_infer();

    EXPECT_TRUE(shape->getOutput()->isShapeValueWhollyFixed());

    // Count operators before folding
    size_t gather_before = 0;
    for (auto op : g->getOperators()) {
        if (op->getOpType() == OpType::Gather)
            gather_before++;
    }
    EXPECT_EQ(gather_before, 1u);

    // Fold should remove Shape and optimize Gather to Constant
    size_t folded = g->foldFixedShapeSubgraph();
    EXPECT_GT(folded, 0u) << "Should fold at least the Shape operator";

    // Verify Gather was optimized away
    size_t gather_after = 0;
    for (auto op : g->getOperators()) {
        if (op->getOpType() == OpType::Gather)
            gather_after++;
    }
    EXPECT_EQ(gather_after, 0u) << "Gather should be replaced with Constant";

    // Verify the consumer now reads from a constant
    auto consumerInput = consumer->getInputs()[0];
    EXPECT_TRUE(consumerInput->isWeight());
    auto result = consumerInput->copyout<int64_t>();
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 224) << "Constant should have the extracted value";

    EXPECT_TRUE(g->checkValid());
}

/// Multiple Gathers on the same constant should all be optimized
TEST(ShapeFold, MultipleGathersOnConstantAllOptimized) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto input = g->addTensor(Shape{1, 3, 224, 224}, DataType::Float32);
    input->setDimDescs({{false, ""}, {false, ""}, {false, ""}, {false, ""}});

    auto shape = g->addOp<ShapeObj>(input, nullptr);
    g->shape_infer();

    // Shape output is fully fixed [1, 3, 224, 224]
    EXPECT_TRUE(shape->getOutput()->isShapeValueWhollyFixed());

    // Create multiple Gathers
    auto g1 =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 1), nullptr, 0);
    auto g2 =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 2), nullptr, 0);
    auto g3 =
        g->addOp<GatherObj>(shape->getOutput(), scalarOf(g, 3), nullptr, 0);
    g->shape_infer();

    // Add consumers to track results
    auto c1 = g->addOp<IdentityObj>(g1->getOutput(), nullptr);
    auto c2 = g->addOp<IdentityObj>(g2->getOutput(), nullptr);
    auto c3 = g->addOp<IdentityObj>(g3->getOutput(), nullptr);
    g->shape_infer();

    size_t gather_before = 0;
    for (auto op : g->getOperators()) {
        if (op->getOpType() == OpType::Gather)
            gather_before++;
    }
    EXPECT_EQ(gather_before, 3u);

    size_t folded = g->foldFixedShapeSubgraph();
    EXPECT_GT(folded, 0u);

    size_t gather_after = 0;
    for (auto op : g->getOperators()) {
        if (op->getOpType() == OpType::Gather)
            gather_after++;
    }
    EXPECT_EQ(gather_after, 0u) << "All Gathers should be optimized";

    // Verify values through consumers
    EXPECT_EQ(c1->getInputs()[0]->copyout<int64_t>()[0], 3);
    EXPECT_EQ(c2->getInputs()[0]->copyout<int64_t>()[0], 224);
    EXPECT_EQ(c3->getInputs()[0]->copyout<int64_t>()[0], 224);

    EXPECT_TRUE(g->checkValid());
}

TEST(ShapeFold, ConstantGatherPreservesGraphOutput) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto gather =
        g->addOp<GatherObj>(constantOf(g, {3, 7}), scalarOf(g, -1), nullptr, 0);
    auto output = gather->getOutput();
    output->setOutput();
    EXPECT_EQ(g->foldFixedShapeSubgraph(), 0u);
    EXPECT_EQ(output->getSource(), gather);
    EXPECT_EQ(g->getOperators().size(), 1u);
    ASSERT_TRUE(g->checkValid());
    g->dataMalloc();
    runtime->run(g);
    EXPECT_EQ(output->copyout<int64_t>(), (vector<int64_t>{7}));
}

TEST(ShapeFold, ConstantGatherAcceptsInt32Index) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto index = g->addTensor({}, DataType::Int32);
    index->setShapeValue({1});
    index->dataMalloc();
    index->copyin(vector<int32_t>{1});
    index->setWeight();
    auto gather = g->addOp<GatherObj>(constantOf(g, {3, 7}), index, nullptr, 0);
    gather->getOutput()->setOutput();
    ASSERT_NO_THROW(g->foldFixedShapeSubgraph());
    ASSERT_TRUE(g->checkValid());
    g->dataMalloc();
    runtime->run(g);
    EXPECT_EQ(gather->getOutput()->copyout<int64_t>(), (vector<int64_t>{7}));
}

TEST(ShapeFold, PartialFoldDoesNotGrowAndReportsNetRemoval) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor({2, 3, 4}, DataType::Float32);
    input->setInput();
    input->setDimDescs({{true, "batch"}, {false, ""}, {false, ""}});
    auto shape = g->addOp<ShapeObj>(input, nullptr)->getOutput();
    auto flat = g->addOp<FlattenObj>(input, nullptr, 0)->getOutput();
    auto output = g->addOp<ReshapeObj>(flat, shape, nullptr)->getOutput();
    output->setOutput();
    g->shape_infer();

    const auto before = g->getOperators().size();
    const auto dropped = g->foldFixedShapeSubgraph();
    const auto after = g->getOperators().size();
    std::cout << "partial fold: before=" << before << " after=" << after
              << " reported_dropped=" << dropped << std::endl;
    EXPECT_LE(after, before);
    EXPECT_EQ(static_cast<int64_t>(before) - static_cast<int64_t>(after),
              static_cast<int64_t>(dropped));
    const auto again = g->foldFixedShapeSubgraph();
    std::cout << "second fold: after=" << g->getOperators().size()
              << " reported_dropped=" << again << std::endl;
    EXPECT_EQ(again, 0u);
    EXPECT_EQ(g->getOperators().size(), after);
    ASSERT_TRUE(g->checkValid());
    for (int batch : {2, 5, 1, 2}) {
        input->setShape({batch, 3, 4});
        g->shape_infer();
        g->dataMalloc();
        input->setData(IncrementalGenerator());
        runtime->run(g);
        EXPECT_EQ(output->getDims(), (Shape{batch, 3, 4}));
        EXPECT_TRUE(output->equalData(input));
    }
}

TEST(ShapeFold, PartialInt32ValuesKeepTypeAndData) {
    auto runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    auto data = g->addTensor({2, 3, 4}, DataType::Float32);
    data->setInput();
    auto requested = g->addTensor({3}, DataType::Int32);
    requested->setInput();
    requested->setShapeValue({2, 3, 4}, {false, true, true});
    auto shape = g->addOp<IdentityObj>(requested, nullptr)->getOutput();
    auto reshape = g->addOp<ReshapeObj>(data, shape, nullptr);
    reshape->getOutput()->setOutput();
    g->shape_infer();
    ASSERT_NO_THROW(g->foldFixedShapeSubgraph());
    ASSERT_TRUE(g->checkValid());
    for (const auto &op : g->getOperators()) {
        if (op->getOpType() == OpType::Concat) {
            for (const auto &input : op->getInputs())
                EXPECT_EQ(input->getDType(), op->getOutput()->getDType());
        }
    }
    for (int batch : {2, 4, 1}) {
        data->setShape({batch, 3, 4});
        requested->setShapeValue({batch, 3, 4}, {false, true, true});
        g->shape_infer();
        g->dataMalloc();
        data->setData(IncrementalGenerator());
        requested->copyin(vector<int32_t>{batch, 3, 4});
        runtime->run(g);
        EXPECT_EQ(reshape->getInputs(1)->copyout<int32_t>(),
                  (vector<int32_t>{batch, 3, 4}));
        EXPECT_TRUE(reshape->getOutput()->equalData(data));
    }
}

} // namespace infini
