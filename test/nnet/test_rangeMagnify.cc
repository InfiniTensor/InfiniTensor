#include "nnet/Visitor/InputVisitor.h"
#include "nnet/Visitor/RangeMagnifyVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(RangeMagnify, Conv5x5) {
    int N = 1, H = 224, W = 224, C = 16, F = 64;
    int R = 5, S = 5;
    auto n = make_ref<VarNode>("n");
    auto c = make_ref<VarNode>("c");
    auto h = make_ref<VarNode>("h");
    auto w = make_ref<VarNode>("w");
    auto f = make_ref<VarNode>("f");
    auto r = make_ref<VarNode>("r");
    auto s = make_ref<VarNode>("s");
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, 0, R / 2, S / 2});
    auto K = make_ref<TensorNode>("K", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {n, c, h + r - R / 2, w + s - S / 2});
    auto subK = makeSubscript(K, {f, c, r, s});

    auto range =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);
    // cout << range->toReadable() << endl;
    auto ret = RangeMagnifyVisitor().magnify(
        range, {{c, {0, C}}, {r, {0, R + 1}}, {s, {0, S + 1}}});
    ASSERT_TRUE(ret);
    EXPECT_EQ(ret->getRange(r), pair(0, R + 1));
    auto inputs = InputVisitor().getInputs(ret);
    ASSERT_EQ(inputs.size(), 2u);
    EXPECT_EQ(inputs[0]->getPadding(0), 0);
    EXPECT_EQ(inputs[0]->getPadding(1), 0);
    EXPECT_EQ(inputs[0]->getPadding(2), 3);
    EXPECT_EQ(inputs[0]->getPadding(3), 3);
    EXPECT_EQ(inputs[1]->getPadding(0), 0);
    EXPECT_EQ(inputs[1]->getPadding(1), 0);
    EXPECT_EQ(inputs[1]->getPadding(2), 1);
    EXPECT_EQ(inputs[1]->getPadding(3), 1);
}
