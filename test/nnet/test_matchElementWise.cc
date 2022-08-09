#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/iterator_table.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

TEST(MatchElementWise, NoMatch) {
    int N = 8, H = 224, W = 224, C = 16, F = 32;
    int R = 9, S = 9;
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

    auto subA = makeSubscript(A, {n, c, h + r, w + s});
    auto subK = makeSubscript(K, {f, c, r + R / 2, s + S / 2});

    auto range = makeRangeOperator(
        {{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
        {{c, {0, C}}, {r, {-R / 2, R / 2 + 1}}, {s, {-S / 2, S / 2 + 1}}},
        subA * subK);
    // cout << range->toReadable() << endl;

    // Derivation
    Formula conv_9x9(range, 0);
    Derivator derivator(1);
    derivator.search(conv_9x9, 1);
    bool hasMatch = false;
    for (const auto &formula : derivator.getCandidates()) {
        if (CountRoutineVisitor().match(formula.root, 0, 0, 1))
            hasMatch = true;
    }
    // Cannot be matched by a single membound
    EXPECT_FALSE(hasMatch);
    derivator.print();
}

TEST(MatchElementWise, TwoStagesWithPadding) {
    int N = 8;
    auto n = make_ref<VarNode>("n");
    auto c = make_ref<VarNode>("c");
    auto h = make_ref<VarNode>("h");
    auto w = make_ref<VarNode>("w");
    auto f = make_ref<VarNode>("f");
    auto r = make_ref<VarNode>("r");
    auto s = make_ref<VarNode>("s");
    auto A =
        make_ref<TensorNode>("A", vector<int>({N, N}), vector<int>{0, N / 2});
    auto K = make_ref<TensorNode>("K", vector<int>({N, N}));

    auto innerSub = makeSubscript(A, {n, h});
    auto innerRange =
        makeRangeOperator({{n, {0, N}}, {h, {0, N}}}, {}, innerSub);
    innerRange->setPaddings({0, 2});
    auto outerSub = makeSubscript(innerRange, {r, s + r});
    auto outerRange =
        makeRangeOperator({{r, {0, 4}}, {s, {0, 5}}}, {}, outerSub);
    // cout << range->toReadable() << endl;

    // Derivation
    Formula conv_9x9(outerRange, 0);
    Derivator derivator;
    derivator.ruleBasedDFS(conv_9x9, 0, {6});
    EXPECT_EQ(derivator.getNumCandidates(), 1);
}

TEST(MatchElementWise, TwoStagesWithImperfectedNestedPadding) {
    int N = 8;
    auto n = make_ref<VarNode>("n");
    auto c = make_ref<VarNode>("c");
    auto h = make_ref<VarNode>("h");
    auto w = make_ref<VarNode>("w");
    auto f = make_ref<VarNode>("f");
    auto r = make_ref<VarNode>("r");
    auto s = make_ref<VarNode>("s");
    auto A = make_ref<TensorNode>("A", vector<int>({100, 100}),
                                  vector<int>{0, N / 2});
    auto K = make_ref<TensorNode>("K", vector<int>({100, 100}));

    auto innerSub = makeSubscript(A, {n, h + n});
    auto innerRange =
        makeRangeOperator({{n, {0, 8}}, {h, {0, 8}}}, {}, innerSub);
    innerRange->setPaddings({0, 2});
    auto outerSub = makeSubscript(innerRange, {r, s + r});
    auto outerRange =
        makeRangeOperator({{r, {0, 4}}, {s, {0, 5}}}, {}, outerSub);
    // cout << range->toReadable() << endl;

    // Derivation
    Formula conv_9x9(outerRange, 0);
    Derivator derivator;
    derivator.ruleBasedDFS(conv_9x9, 0, {6});
    EXPECT_EQ(derivator.getNumCandidates(), 0);
}