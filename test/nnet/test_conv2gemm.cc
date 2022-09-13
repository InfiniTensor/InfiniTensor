#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/test.h"
#include "gtest/gtest.h"
#include <chrono>
using namespace nnet;
using namespace std;

TEST(Conv2gemm, NCHW_FCRS_ruleBased) {
    // A[n,h+r,w+s,c]*K[r,s,f,c]
    const int N = 8, H = 224, W = 224, C = 16, F = 32, R = 3, S = 3;
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, 0, R / 2, S / 2});
    auto K = make_ref<TensorNode>("K", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {n, c, h + r - R / 2, w + s - S / 2});
    auto subK = makeSubscript(K, {f, c, r, s});

    auto range =
        makeRangeOperator({{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);
    // cout << range->toReadable() << endl;

    // Derivation
    Formula conv_1x1_nhwc_fc(range, 0);
    Derivator derivator(12);
    // const vector<int> rules = {3, 2, 2, 5, 2, 2, 6, 6};
    const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
    // derivator.ruleBasedDFS(conv_1x1_nhwc_fc, 0, rules);
    derivator.search(conv_1x1_nhwc_fc, 0);
    // Stage merge with padding is not realized
    EXPECT_EQ(derivator.getSearchedMaxDepth(), 5);
    ASSERT_GT(derivator.getNumCandidates(), 0);
    derivator.print();
    bool hasMatch = false;
    for (const auto &formula : derivator.getCandidates()) {
        if (CountRoutineVisitor().match(formula.root, 1, 0, 3))
            hasMatch = true;
    }
    EXPECT_TRUE(hasMatch);
}

TEST(Conv2gemm, NHWC_RSFC_ruleBased) {
    // A[n,h+r,w+s,c]*K[r,s,f,c]
    const int N = 8, H = 224, W = 224, C = 16, F = 32, R = 3, S = 3;
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, C}),
                                  vector<int>{0, R / 2, S / 2, 0});
    // auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, h + r - R / 2, w + s - S / 2, c});
    auto subK = makeSubscript(K, {r, s, f, c});

    auto range =
        makeRangeOperator({{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);
    // cout << range->toReadable() << endl;

    // Derivation
    Formula conv_1x1_nhwc_fc(range, 0);
    Derivator derivator(5);
    // const vector<int> rules = {3, 2, 2, 5, 2, 2, 6, 6};
    const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
    // derivator.ruleBasedDFS(conv_1x1_nhwc_fc, 0, rules);
    derivator.search(conv_1x1_nhwc_fc, 0);
    // Stage merge with padding is not realized
    EXPECT_EQ(derivator.getSearchedMaxDepth(), 5);
    EXPECT_GE(derivator.getNumCandidates(), 1);
    derivator.print();
    derivator.printStatistics();
}

TEST(Conv2gemm, Derivation_dfs) {
    // A[n,h+r,w+s,c]*K[r,s,f,c]
    int N = 8, H = 224, W = 224, C = 16, F = 32;
    int R = 3, S = 3;
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, C}),
                                  vector<int>{0, R / 2, S / 2, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, h + r - R / 2, w + s - S / 2, c});
    auto subK = makeSubscript(K, {r, s, f, c});

    auto range =
        makeRangeOperator({{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);
    // cout << range->toReadable() << endl;

    // Derivation
    Formula conv_1x1_nhwc_fc(range, 0);
    Derivator derivator(12);
    derivator.search(conv_1x1_nhwc_fc, 0);
    EXPECT_GT(derivator.getNumCandidates(), 0);
    derivator.print();
}

void Conv2gemm_NHWC_RSFC_search(int maxDepth, bool enalbeHashPruning) {
    // A[n,h+r,w+s,c]*K[r,s,f,c]
    int N = 1, H = 7, W = 7, C = 32, F = 32, R = 3, S = 3;
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, C}),
                                  vector<int>{0, R / 2, S / 2, 0});
    // auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, h + r - R / 2, w + s - S / 2, c});
    auto subK = makeSubscript(K, {r, s, f, c});

    auto range =
        makeRangeOperator({{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);
    // cout << ange->toReadable() << endl;

    // Derivation
    Formula conv_1x1_nhwc_fc(range, 0);
    Derivator derivator(maxDepth, enalbeHashPruning);
    // const vector<int> rules = {3, 2, 2, 5, 2, 2, 6, 6};
    const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
    // derivator.ruleBasedDFS(conv_1x1_nhwc_fc, 0, rules);
    derivator.search(conv_1x1_nhwc_fc, 0);
    // Stage merge with padding is not realized
    EXPECT_EQ(derivator.getSearchedMaxDepth(), maxDepth);
    EXPECT_GE(derivator.getNumCandidates(), 1);
    derivator.printStatistics();
}

TEST(Conv2gemm, timing_NHWC_RSFC_search) {
    for (bool enalbeHashPruning : {true, false}) {
        // Disabled to pass 10s time limit
        for (int maxDepth = 5; maxDepth < 5; ++maxDepth) {
            printf("Max depth = %d, Hash = %d\n", maxDepth, enalbeHashPruning);
            auto t_start = std::chrono::high_resolution_clock::now();
            Conv2gemm_NHWC_RSFC_search(maxDepth, enalbeHashPruning);
            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed_time_s =
                std::chrono::duration<double, std::milli>(t_end - t_start)
                    .count() /
                1000;
            printf("Elapsed time (s) = %lf\n", elapsed_time_s);
        }
    }
}

// Conv2gemm requires thorough update, this is disabled temporarily
TEST(Conv2gemm, CheckCorrectness) {
    const string fnPrefix = "../test/nnet/log/conv2gemm/Conv2gemm_NCHW_RSFC_";
    // conv2gemm_7 has T3
    EXPECT_TRUE(checkExprLogSame(fnPrefix, 0, 7));
}

TEST(Conv2gemm, NCHW_RSFC_search) {
    // A[n,h+r,w+s,c]*K[r,s,f,c]
    const int N = 1, H = 7, W = 7, C = 512, F = 512, R = 3, S = 3;
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, C}),
                                  vector<int>{0, R / 2, S / 2, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, h + r - R / 2, w + s - S / 2, c});
    auto subK = makeSubscript(K, {r, s, f, c});

    auto range =
        makeRangeOperator({{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);

    // Derivation
    Formula conv_3x3_nhwc_rsfc(range, 0);
    Derivator derivator(10);

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
        // derivator.setDumpFirstSuccess("Conv2gemm_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_3x3_nhwc_rsfc, 0, rules);
    } else {
        derivator.search(conv_3x3_nhwc_rsfc, 0);
    }

    ASSERT_GE(derivator.getNumCandidates(), 1);
    int nMatches = matchExprResult(
        derivator, "../test/nnet/log/conv2gemm/Conv2gemm_NCHW_RSFC_11.expr");
    EXPECT_GE(nMatches, 1);
    // derivator.print();
    derivator.printStatistics();
}

TEST(Conv2gemm1x1, NHWC_RSFC_ruleBased) {
    // A[n,h+r,w+s,c]*K[r,s,f,c]
    const int N = 1, H = 7, W = 7, C = 512, F = 512, R = 1, S = 1;
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, C}));
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, h + r, w + s, c});
    auto subK = makeSubscript(K, {r, s, f, c});

    auto range =
        makeRangeOperator({{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);

    Formula conv_1x1_nhwc_fc(range, 0);
    Derivator derivator(7);
    const vector<int> rules = {3, 2, 2, 8, 8, 6, 6};
    derivator.ruleBasedDFS(conv_1x1_nhwc_fc, 0, rules);
    ASSERT_GT(derivator.getNumCandidates(), 0);
    derivator.printStatistics();
    bool hasMatch = false;
    for (const auto &formula : derivator.getCandidates()) {
        if (CountRoutineVisitor().match(formula.root, 1, 0, 3))
            hasMatch = true;
    }
    EXPECT_TRUE(hasMatch);
}

TEST(Conv2gemm1x1, NCHW_FCRS_search) {
    // A[n,h+r,w+s,c]*K[f,c,r,s]
    const int N = 1, H = 7, W = 7, C = 512, F = 512, R = 1, S = 1;
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, R / 2, S / 2, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {n, c, h + r - R / 2, w + s - S / 2});
    auto subK = makeSubscript(K, {f, c, r, s});

    auto range =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);

    // Derivation
    Formula conv_3x3_nhwc_rsfc(range, 0);
    Derivator derivator(10);

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules = {3, 2, 2, 8, 8, 6, 6};
        // derivator.setDumpFirstSuccess("Conv2gemm_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_3x3_nhwc_rsfc, 0, rules);
    } else {
        derivator.search(conv_3x3_nhwc_rsfc, 0);
    }

    ASSERT_GE(derivator.getNumCandidates(), 1);
}

TEST(Conv2gemm1x7, NCHW_FCRS_search) {
    const int N = 1, C = 2048, H = 7, W = 7, F = 128, R = 1,
              S = 7; // gcn_Conv_137
    DEFINE_VAR(n, c, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, 0, R / 2, S / 2});
    auto K = make_ref<TensorNode>("K", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {n, c, h + r - R / 2, w + s - S / 2});
    auto subK = makeSubscript(K, {f, c, r, s});

    auto range =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);

    // Derivation
    Formula conv_1x7(range, 0);
    Derivator derivator(10, true, nnet::Derivator::LogMode::NoLog);

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
        derivator.setDumpFirstSuccess("Conv2gemm_1x7_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_1x7, 0, rules);
    } else {
        derivator.search(conv_1x7, 0);
    }

    ASSERT_GE(derivator.getNumCandidates(), 1);
    int nMatches = matchExprResult(
        derivator,
        "../test/nnet/log/conv2gemm_1x7/Conv2gemm_1x7_NCHW_FCRS_11.expr");
    EXPECT_GE(nMatches, 1);
}