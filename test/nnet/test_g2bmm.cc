#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/test.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

TEST(GBMM, RuleBased) {
    const int n_heads = 8, seq_len = 10000, feat_len = 512;
    const int Batch = n_heads, M = seq_len, K = feat_len, W = 32, dilation = 4;
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(n);
    DEFINE_VAR(w);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, 2 * W + 1}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, M, K}),
                                  vector<int>{0, dilation * W, 0});
    auto subA = makeSubscript(A, {b, m, w});
    // auto subB = makeSubscript(B, {b, m + dilation * (w - W), n});
    auto subB = makeSubscript(B, {b, m + dilation * w - dilation * W, n});
    auto range = makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {n, {0, K}}},
                                   {{w, {0, 2 * W + 1}}}, subA * subB);
    dbg(range);

    // Derivation: this work without padding check in stage merging
    Formula dialted_g2bmm(range, 0);
    Derivator derivator;

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules{1, 7, 7, 2, 8, 6, 6};
        derivator.setDumpFirstSuccess("GBMM_");
        derivator.ruleBasedDFS(dialted_g2bmm, 0, rules);
    } else {
        derivator.search(dialted_g2bmm, 0);
    }

    ASSERT_GE(derivator.getNumCandidates(), 1);
    int nMatches =
        matchExprResult(derivator, "../test/nnet/log/gbmm/GBMM_9.expr");
    EXPECT_GE(nMatches, 1);
    derivator.print();
    derivator.printStatistics();
}

TEST(G2BMM, RuleBased) {
    const int n_heads = 8, seq_len = 10000, feat_len = 512;
    const int Batch = n_heads, M = seq_len, K = feat_len, W = 32, dilation = 4;
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, M, K}),
                                  vector<int>{0, dilation * W, 0});

    auto subA = makeSubscript(A, {b, m, k});
    auto subB = makeSubscript(B, {b, m + dilation * (w - W), k});
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {w, {0, 2 * W + 1}}},
                          {{k, {0, K}}}, subA * subB);

    // Derivation: this work without padding check in stage merging
    Formula dialted_g2bmm(range, 0);
    Derivator derivator;

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules{1, 7, 7, 2, 8, 6, 6};
        derivator.setDumpFirstSuccess("G2BMM_");
        derivator.ruleBasedDFS(dialted_g2bmm, 0, rules);
    } else {
        derivator.search(dialted_g2bmm, 0);
    }

    ASSERT_GE(derivator.getNumCandidates(), 1);
    int nMatches =
        matchExprResult(derivator, "../test/nnet/log/g2bmm/G2BMM_9.expr");
    EXPECT_GE(nMatches, 1);
    derivator.print();
    derivator.printStatistics();
}
