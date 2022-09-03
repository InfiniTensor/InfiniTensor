#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/test.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

TEST(Conv2conv, 9x9_NCHW_FCRS) {
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

    auto subA = makeSubscript(A, {n, c, h + r - R / 2, w + s - S / 2});
    auto subK = makeSubscript(K, {f, c, r, s});

    auto range =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);

    Formula conv_9x9(range, 0);
    Derivator derivator(8);

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules{1, 1, 3, 2, 2, 5, 8, 8, 6, 90};
        // derivator.setDumpFirstSuccess("Conv2conv_9x9_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_9x9, 0, rules, {}, true);
    } else
        derivator.search(conv_9x9, 0);

    int nMatches = matchExprResult(
        derivator,
        "../test/nnet/log/conv2conv/Conv2conv_9x9_NCHW_FCRS_14.expr");
    derivator.print();
    derivator.printStatistics();
    EXPECT_GE(nMatches, 1);
}

TEST(Conv2conv, 6x6_RuleBased_NCHW_FCRS) {
    int N = 1, H = 224, W = 224, C = 16, F = 64;
    int R = 6, S = 6;
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

    auto subA =
        makeSubscript(A, {n, c, h + r - (R - 1) / 2, w + s - (S - 1) / 2});
    auto subK = makeSubscript(K, {f, c, r, s});

    auto range =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);

    Formula conv_6x6(range, 0);
    Derivator derivator;

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules{1, 1, 3, 2, 2, 5, 8, 8, 6, 6};
        // derivator.setDumpFirstSuccess("Conv2conv_6x6_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_6x6, 0, rules, {}, true);
    } else
        derivator.search(conv_6x6, 0);

    ASSERT_GE(derivator.getNumCandidates(), 1);
    int nMatches = matchExprResult(
        derivator,
        "../test/nnet/log/conv2conv/Conv2conv_6x6_NCHW_FCRS_14.expr");
    derivator.print();
    derivator.printStatistics();
    EXPECT_GE(nMatches, 1);
}

TEST(Conv2conv, 5x5_RuleBased_NCHW_FCRS) {
    int N = 16, C = 32, H = 224, W = 224, F = 1;
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

    Formula conv_9x9(range, 0);
    Derivator derivator(7);

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules{9, 1, 1, 3, 2, 2, 5, 8, 8, 6, 6};
        derivator.setDumpFirstSuccess("Conv2conv_5x5_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_9x9, 0, rules, {}, true);
    } else
        derivator.search(conv_9x9, 0);

    int nMatches = matchExprResult(
        derivator,
        "../test/nnet/log/conv2conv/Conv2conv_5x5_NCHW_FCRS_15.expr");
    derivator.print();
    derivator.printStatistics();
    EXPECT_GE(nMatches, 1);
}