#include "core/blob.h"
#include "core/graph_match.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/extend.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce.h"
#include "operators/slice.h"
#include "operators/split.h"
#include "operators/unary.h"
#include "test.h"
namespace infini {
// hrnet48 head   match conv-relu
TEST(SubGraphRewriter, subGraphMatch1) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);
    Tensor i0 = g->addTensor({1, 3, 244, 244}, DataType::UInt32);
    Tensor w0 = g->addTensor({64, 3, 3, 3}, DataType::UInt32);
    auto conv = g->addOp<ConvObj>(i0, w0, nullptr);
    auto relu = g->addOp<ReluObj>(conv->getOutput(), nullptr);

    auto w1 = g->addTensor({64, 64, 3, 3}, DataType::UInt32);
    auto conv1 = g->addOp<ConvObj>(relu->getOutput(0), w1, nullptr);
    auto relu1 = g->addOp<ReluObj>(conv1->getOutput(), nullptr);

    auto w2 = g->addTensor({64, 64, 1, 1}, DataType::UInt32);
    auto conv2 = g->addOp<ConvObj>(relu1->getOutput(0), w2, nullptr);
    auto relu2 = g->addOp<ReluObj>(conv2->getOutput(), nullptr);

    auto w3 = g->addTensor({256, 64, 1, 1}, DataType::UInt32);
    auto conv3 = g->addOp<ConvObj>(relu1->getOutput(0), w3, nullptr);

    auto w4 = g->addTensor({64, 64, 3, 3}, DataType::UInt32);
    auto conv4 = g->addOp<ConvObj>(relu2->getOutput(0), w4, nullptr);
    auto relu4 = g->addOp<ReluObj>(conv4->getOutput(), nullptr);

    Tensor si0 =
        make_ref<TensorObj>(Shape{1, 64, 112, 112}, DataType::UInt32, runtime);
    SubGraph subG = make_ref<SubGraphObj>(runtime, TensorVec{si0});
    Tensor sw0 = subG->addTensor({64, 64, 3, 3}, DataType::UInt32);
    auto sconv0 = subG->addOp<ConvObj>(si0, sw0, nullptr);
    auto srelu0 = subG->addOp<ReluObj>(sconv0->getOutput(), nullptr);
    subG->setOutputs(srelu0->getOutputs());

    SubGraphRewriter v(g);
    vector<MatchGraph> subgs = v.findMatch(subG);

    EXPECT_TRUE(subgs.size() == 2);
}

TEST(MatchGraph, single_input) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    // subG0
    Tensor si0 =
        make_ref<TensorObj>(Shape{1, 96, 28, 28}, DataType::UInt32, runtime);
    SubGraph subG = make_ref<SubGraphObj>(runtime, TensorVec{si0});
    {
        auto srelu0 = subG->addOp<ReluObj>(si0, nullptr);
        auto sw0 = subG->addTensor({96, 96, 3, 3}, DataType::UInt32);
        auto sconv0 = subG->addOp<ConvObj>(srelu0->getOutput(0), sw0, nullptr);
        auto srelu1 = subG->addOp<ReluObj>(sconv0->getOutput(), nullptr);
        auto sw1 = subG->addTensor({96, 96, 3, 3}, DataType::UInt32);
        auto sconv1 = subG->addOp<ConvObj>(srelu1->getOutput(0), sw1, nullptr);
        auto sadd0 = subG->addOp<AddObj>(sconv1->getOutput(0),
                                         srelu0->getOutput(0), nullptr);
        subG->setOutputs({sadd0->getOutput()});
    }
    // subG1
    Tensor si00 =
        make_ref<TensorObj>(Shape{1, 48, 56, 56}, DataType::UInt32, runtime);
    SubGraph subG1 = make_ref<SubGraphObj>(runtime, TensorVec{si00});
    {
        auto srelu0 = subG1->addOp<ReluObj>(si00, nullptr);
        auto sw0 = subG1->addTensor({48, 48, 3, 3}, DataType::UInt32);
        auto sconv0 = subG1->addOp<ConvObj>(srelu0->getOutput(0), sw0, nullptr);
        auto srelu1 = subG1->addOp<ReluObj>(sconv0->getOutput(), nullptr);
        auto sw1 = subG1->addTensor({48, 48, 3, 3}, DataType::UInt32);
        auto sconv1 = subG1->addOp<ConvObj>(srelu1->getOutput(0), sw1, nullptr);
        auto sadd0 = subG1->addOp<AddObj>(sconv1->getOutput(0),
                                          srelu0->getOutput(0), nullptr);
        subG1->setOutputs({sadd0->getOutput()});
    }

    // graph
    Graph g = make_ref<GraphObj>(runtime);
    SubGraphRewriter v(g);

    Tensor i0 = g->addTensor({1, 256, 56, 56}, DataType::UInt32);
    auto relu0 = g->addOp<ReluObj>(i0, nullptr);

    Tensor w0 = g->addTensor({96, 256, 3, 3}, DataType::UInt32);
    auto conv0 = g->addOp<ConvObj>(relu0->getOutput(0), w0, nullptr, 1, 1,
                                   nullptr, 2, 2);

    auto o0 = v.addSubGraph(subG, {conv0->getOutput(0)});
    auto o1 = v.addSubGraph(subG, o0);
    auto o2 = v.addSubGraph(subG, o1);
    auto o3 = v.addSubGraph(subG, o2);
    auto relu4 = g->addOp<ReluObj>(o3[0], nullptr);

    Tensor w10 = g->addTensor({48, 256, 3, 3}, DataType::UInt32);
    auto conv10 = g->addOp<ConvObj>(relu0->getOutput(0), w10, nullptr);
    auto o10 = v.addSubGraph(subG1, {conv10->getOutput(0)});
    auto o11 = v.addSubGraph(subG1, o10);
    auto o12 = v.addSubGraph(subG1, o11);
    auto o13 = v.addSubGraph(subG1, o12);
    auto relu10 = g->addOp<ReluObj>(o13[0], nullptr);
    Tensor w1 = g->addTensor({96, 48, 3, 3}, DataType::UInt32);
    auto conv1 = g->addOp<ConvObj>(relu10->getOutput(), w1, nullptr, 1, 1,
                                   nullptr, 2, 2);
    auto add1 =
        g->addOp<AddObj>(relu4->getOutput(), conv1->getOutput(), nullptr);

    auto o4 = v.addSubGraph(subG, TensorVec{add1->getOutput(0)});

    EXPECT_EQ(g->getOperators().size(), 52);
    vector<MatchGraph> subgs = v.findMatch(subG);
    EXPECT_TRUE(subgs.size() == 5);

    vector<MatchGraph> subgs1 = v.findMatch(subG1);
    EXPECT_TRUE(subgs1.size() == 4);

    // test replace
    Tensor sii0 =
        make_ref<TensorObj>(Shape{1, 96, 28, 28}, DataType::UInt32, runtime);
    SubGraph subG2 = make_ref<SubGraphObj>(runtime, TensorVec{sii0});
    {
        auto srelu0 = subG2->addOp<ReluObj>(sii0, nullptr);
        auto sw0 = subG2->addTensor({96, 96, 3, 3}, DataType::UInt32);
        auto sconv0 = subG2->addOp<ConvObj>(srelu0->getOutput(0), sw0, nullptr);
        subG2->setOutputs(sconv0->getOutputs());
    }

    v.replaceSubGraph(subG, subG2);
    EXPECT_EQ(g->getOperators().size(), 37);
}

TEST(MatchGraph, multi_input) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    // subG0
    Tensor i0 =
        make_ref<TensorObj>(Shape{3, 4, 5, 2}, DataType::UInt32, runtime);
    Tensor i1 = make_ref<TensorObj>(Shape{3, 4, 5}, DataType::UInt32, runtime);
    SubGraph subG = make_ref<SubGraphObj>(runtime, TensorVec{i0, i1});
    {
        auto reduce0 =
            subG->addOp<ReduceMeanObj>(i0, nullptr, vector<int>{3}, false);
        auto sub0 = subG->addOp<SubObj>(reduce0->getOutput(0), i1, nullptr);
        subG->setOutputs(sub0->getOutputs());
    }

    SubGraph replaceG = make_ref<SubGraphObj>(
        runtime, TensorVec{i0->clone(runtime), i1->clone(runtime)});
    {
        auto reduce0 =
            replaceG->addOp<ReduceMeanObj>(replaceG->getInputsFromOutside()[0],
                                           nullptr, vector<int>{3}, false);
        auto sub0 = replaceG->addOp<AddObj>(reduce0->getOutput(0),
                                            replaceG->getInputsFromOutside()[1],
                                            nullptr);
        replaceG->setOutputs(sub0->getOutputs());
    }

    Graph g = make_ref<GraphObj>(runtime);
    SubGraphRewriter v(g);
    {
        Tensor i0 = g->addTensor({3, 4, 5, 2}, DataType::UInt32);
        Tensor i1 = g->addTensor({3, 4, 5, 2}, DataType::UInt32);
        auto add0 = g->addOp<AddObj>(i0, i1, nullptr);
        auto relu0 = g->addOp<ReluObj>(add0->getOutput(), nullptr);
        auto reduce0 = g->addOp<ReduceMeanObj>(relu0->getOutput(), nullptr,
                                               vector<int>{3}, false);
        auto o0 =
            v.addSubGraph(subG, {add0->getOutput(), reduce0->getOutput()});

        Tensor i2 = g->addTensor({3, 4, 5}, DataType::UInt32);
        auto pow0 = g->addOp<PowObj>(o0[0], i2, nullptr);

        Tensor i3 = g->addTensor({3, 4, 5, 2}, DataType::UInt32);
        auto reduce1 =
            g->addOp<ReduceMeanObj>(i3, nullptr, vector<int>{3}, false);
        auto sub0 = g->addOp<SubObj>(reduce1->getOutput(0), pow0->getOutput(0),
                                     nullptr);

        auto matches = v.findMatch(subG);
        EXPECT_EQ(2, matches.size());

        auto div0 = g->addOp<DivObj>(reduce1->getOutput(0), i2, nullptr);
        auto add1 =
            g->addOp<AddObj>(sub0->getOutput(), div0->getOutput(), nullptr);
        matches = v.findMatch(subG);
        EXPECT_EQ(1, matches.size());

        // two matched subgraphs overlaped,so only replaced one sub graph
        v.replaceSubGraph(subG, replaceG);
        EXPECT_EQ(1, v.findMatch(replaceG).size());
    }
}

TEST(MatchGraph, multi_output) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    // subg0
    Tensor i =
        make_ref<TensorObj>(Shape{1, 192, 71, 71}, DataType::UInt32, runtime);
    SubGraph subg0 = make_ref<SubGraphObj>(runtime, TensorVec{i});
    {
        auto maxpool =
            subg0->addOp<MaxPoolObj>(i, nullptr, 3, 3, 1, 1, 0, 0, 2, 2, 0);
        Tensor w0 = subg0->addTensor(Shape{64, 192, 1, 1}, DataType::UInt32);
        auto conv0 = subg0->addOp<ConvObj>(maxpool->getOutput(0), w0, nullptr);
        auto relu0 = subg0->addOp<ReluObj>(conv0->getOutput(0), nullptr);

        auto avgpool = subg0->addOp<AvgPoolObj>(maxpool->getOutput(0), nullptr,
                                                3, 3, 0, 0, 0, 0, 1, 1, 0);
        subg0->setOutputs(
            TensorVec{relu0->getOutput(0), avgpool->getOutput(0)});
    }

    SubGraph subg1 =
        make_ref<SubGraphObj>(runtime, TensorVec{i->clone(runtime)});
    {
        auto avgpool =
            subg1->addOp<AvgPoolObj>(subg1->getInputsFromOutside()[0], nullptr,
                                     3, 3, 1, 1, 0, 0, 2, 2, 0);

        auto relu0 = subg1->addOp<ReluObj>(avgpool->getOutput(0), nullptr);

        auto split0 =
            subg1->addOp<SplitObj>(avgpool->getOutput(0), std::nullopt, 1, 3);
        subg1->setOutputs(TensorVec{split0->getOutput(1), relu0->getOutput(0)});
    }

    Graph g = make_ref<GraphObj>(runtime);
    SubGraphRewriter v(g);
    {
        auto input = g->cloneTensor(i);
        auto outs = v.addSubGraph(subg0, {input});
        EXPECT_EQ(2, outs.size());
        Tensor w0 = g->addTensor(Shape{96, 64, 3, 3}, DataType::UInt32);
        auto conv0 = g->addOp<ConvObj>(outs[0], w0, nullptr, 1, 1);
        auto relu0 = g->addOp<ReluObj>(conv0->getOutput(0), nullptr);

        Tensor w1 = g->addTensor(Shape{96, 96, 3, 3}, DataType::UInt32);
        auto conv1 = g->addOp<ConvObj>(relu0->getOutput(), w1, nullptr, 1, 1);
        auto relu1 = g->addOp<ReluObj>(conv1->getOutput(0), nullptr);

        Tensor w2 = g->addTensor(Shape{32, 192, 1, 1}, DataType::UInt32);
        auto conv2 = g->addOp<ConvObj>(outs[1], w2, nullptr);
        auto relu2 = g->addOp<ReluObj>(conv2->getOutput(0), nullptr);

        Tensor i0 = g->addTensor(Shape{1, 64, 35, 35}, DataType::UInt32);
        Tensor i1 = g->addTensor(Shape{1, 64, 35, 35}, DataType::UInt32);
        auto concat = g->addOp<ConcatObj>(
            TensorVec{i0, i1, relu1->getOutput(), relu2->getOutput()}, nullptr,
            1);
        auto o = concat->getOutput();
        EXPECT_TRUE((o->getDims() == Shape{1, 256, 35, 35}));
    }

    auto matches = v.findMatch(subg0);
    EXPECT_EQ(1, matches.size());

    v.replaceSubGraph(subg0, subg1);
    auto matches2 = v.findMatch(subg1);
    EXPECT_EQ(1, matches2.size());
}

// gcn
TEST(MatchGraph, multi_input_output) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    // subg0
    Tensor i0 =
        make_ref<TensorObj>(Shape{1, 64, 112, 112}, DataType::UInt32, runtime);
    Tensor i1 =
        make_ref<TensorObj>(Shape{1, 64, 56, 56}, DataType::UInt32, runtime);
    SubGraph subg0 = make_ref<SubGraphObj>(runtime, TensorVec{i0, i1});
    {
        auto slice = subg0->addOp<SliceObj>(i0, nullptr, vector<int>{0, 0},
                                            vector<int>{56, 56},
                                            vector<int>{2, 3}, std::nullopt);
        auto relu0 = subg0->addOp<ReluObj>(slice->getOutput(0), nullptr);
        Tensor w0 = subg0->addTensor(Shape{256, 64, 1, 1}, DataType::UInt32);
        auto conv0 = subg0->addOp<ConvObj>(relu0->getOutput(0), w0, nullptr);

        auto conv1 = subg0->addOp<ConvObj>(i1, w0, nullptr);
        auto add = subg0->addOp<AddObj>(conv0->getOutput(0),
                                        conv1->getOutput(0), nullptr);

        auto relu1 = subg0->addOp<ReluObj>(add->getOutput(0), nullptr);
        Tensor w2 = subg0->addTensor(Shape{128, 256, 1, 1}, DataType::UInt32);
        auto conv2 = subg0->addOp<ConvObj>(relu1->getOutput(0), w2, nullptr);
        auto maxpool = subg0->addOp<MaxPoolObj>(relu1->getOutput(0), nullptr, 3,
                                                3, 1, 1, 0, 0, 2, 2, 0);
        subg0->setOutputs(
            TensorVec{conv2->getOutput(0), maxpool->getOutput(0)});
    }

    SubGraph subg1 = make_ref<SubGraphObj>(runtime, TensorVec{i1, i0});
    {
        auto slice = subg1->addOp<SliceObj>(i0, nullptr, vector<int>{0, 0},
                                            vector<int>{56, 56},
                                            vector<int>{2, 3}, std::nullopt);
        auto relu0 = subg1->addOp<ReluObj>(slice->getOutput(0), nullptr);
        Tensor w0 = subg1->addTensor(Shape{256, 64, 1, 1}, DataType::UInt32);
        auto conv0 = subg1->addOp<ConvObj>(relu0->getOutput(0), w0, nullptr);

        auto conv1 = subg1->addOp<ConvObj>(i1, w0, nullptr);
        auto add = subg1->addOp<AddObj>(conv1->getOutput(0),
                                        conv0->getOutput(0), nullptr);

        auto relu1 = subg1->addOp<ReluObj>(add->getOutput(0), nullptr);
        Tensor w2 = subg1->addTensor(Shape{128, 256, 1, 1}, DataType::UInt32);
        auto conv2 = subg1->addOp<ConvObj>(relu1->getOutput(0), w2, nullptr);
        auto maxpool = subg1->addOp<MaxPoolObj>(relu1->getOutput(0), nullptr, 3,
                                                3, 1, 1, 0, 0, 2, 2, 0);
        subg1->setOutputs(
            TensorVec{maxpool->getOutput(0), conv2->getOutput(0)});
    }

    SubGraph subg2 = make_ref<SubGraphObj>(runtime, TensorVec{i0, i1});
    {
        auto extend = subg2->addOp<ExtendObj>(i0, nullptr, 1, 3);

        auto slice = subg2->addOp<SliceObj>(
            extend->getOutput(0), nullptr, vector<int>{0, 0},
            vector<int>{56, 56}, vector<int>{2, 3}, std::nullopt);

        auto extend1 = subg2->addOp<ExtendObj>(i1, nullptr, 1, 3);
        auto add = subg2->addOp<AddObj>(extend1->getOutput(0),
                                        slice->getOutput(0), nullptr);

        auto relu1 = subg2->addOp<ReluObj>(add->getOutput(0), nullptr);
        Tensor w2 = subg2->addTensor(Shape{128, 256, 1, 1}, DataType::UInt32);
        auto conv2 = subg2->addOp<ConvObj>(relu1->getOutput(0), w2, nullptr);
        auto avgpool = subg2->addOp<AvgPoolObj>(relu1->getOutput(0), nullptr, 3,
                                                3, 1, 1, 0, 0, 2, 2, 0);
        subg2->setOutputs(
            TensorVec{conv2->getOutput(0), avgpool->getOutput(0)});
    }

    Graph g = make_ref<GraphObj>(runtime);
    SubGraphRewriter v(g);
    {
        auto i = g->addTensor(Shape{1, 64, 112, 112}, DataType::UInt32);
        auto relu = g->addOp<ReluObj>(i, nullptr);
        auto maxPool = g->addOp<MaxPoolObj>(relu->getOutput(0), nullptr, 3, 3,
                                            1, 1, 1, 1, 2, 2, 0);
        auto out0 =
            v.addSubGraph(subg0, {relu->getOutput(0), maxPool->getOutput(0)});
        auto out1 =
            v.addSubGraph(subg1, {maxPool->getOutput(0), relu->getOutput(0)});
        EXPECT_EQ(2, out0.size());
        EXPECT_EQ(2, out1.size());
        auto div = g->addOp<DivObj>(out0[0], out1[1], nullptr);
        auto sub = g->addOp<SubObj>(out0[1], out1[0], nullptr);
    }

    EXPECT_EQ(2, v.findMatch(subg0).size());
    EXPECT_EQ(2, v.findMatch(subg1).size());
    v.replaceSubGraph(subg0, subg2);
    EXPECT_EQ(v.findMatch(subg2).size(), 2);
}

/* One Node having two or more successors is not supported yet.
TEST(MatchGraph, same_successor) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    // subg0
    Tensor i0 =
        make_ref<TensorObj>(Shape{1, 64, 112, 112}, DataType::UInt32, runtime);
    Tensor i1 =
        make_ref<TensorObj>(Shape{1, 64, 112, 112}, DataType::UInt32, runtime);
    SubGraph subg0 = make_ref<SubGraphObj>(runtime, TensorVec{i0, i1});
    {
        auto add0 = subg0->addOp<AddObj>(i0, i1, nullptr);
        auto add1 = subg0->addOp<AddObj>(add0->getOutput(0), i1, nullptr);
        auto add2 = subg0->addOp<AddObj>(add0->getOutput(0), i1, nullptr);

        auto mul = subg0->addOp<MulObj>(add1->getOutput(0), i1, nullptr);
        auto div = subg0->addOp<DivObj>(add2->getOutput(0), i1, nullptr);

        auto sub =
            subg0->addOp<SubObj>(mul->getOutput(0), div->getOutput(0), nullptr);

        subg0->setOutputs(TensorVec{sub->getOutput(0)});
    }

    // pattern
    SubGraph pattern1 = make_ref<SubGraphObj>(runtime, TensorVec{i0, i1});
    {
        auto add0 = pattern1->addOp<AddObj>(i0, i1, nullptr);
        auto add1 = pattern1->addOp<AddObj>(add0->getOutput(0), i1, nullptr);
        auto div = pattern1->addOp<DivObj>(add1->getOutput(0), i1, nullptr);
        pattern1->setOutputs(TensorVec{add0->getOutput(0), div->getOutput(0)});
    }

    // pattern
    SubGraph pattern2 = make_ref<SubGraphObj>(runtime, TensorVec{i0, i1});
    {
        auto add0 = pattern2->addOp<AddObj>(i0, i1, nullptr);
        auto add1 = pattern2->addOp<AddObj>(add0->getOutput(0), i1, nullptr);
        pattern2->setOutputs(TensorVec{add0->getOutput(0), add1->getOutput(0)});
    }

    Graph g = make_ref<GraphObj>(runtime);
    SubGraphRewriter v(g);
    {
        i0 = g->addTensor(Shape{1, 64, 112, 112}, DataType::UInt32);
        i1 = g->addTensor(Shape{1, 64, 112, 112}, DataType::UInt32);
        auto out0 = v.addSubGraph(subg0, {i0, i1});
    }

    EXPECT_EQ(1, v.findMatch(pattern1).size());
    EXPECT_EQ(2, v.findMatch(pattern2).size());
    v.replaceSubGraph(pattern2, pattern1);
    EXPECT_EQ(v.findMatch(pattern2).size(), 2);
}*/
} // namespace infini
