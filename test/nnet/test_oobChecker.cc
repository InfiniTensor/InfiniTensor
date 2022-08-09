#include "nnet/Visitor/CheckOOBVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(OOB, noOOB) {
    const int n_heads = 8, seq_len = 10000, feat_len = 512;
    // dilation_heads = 2;
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

    CheckOOBVisitor oobchecker;
    EXPECT_FALSE(oobchecker.checkRangeOp(range));
}

TEST(OOB, hasOOB) {
    const int n_heads = 8, seq_len = 10000, feat_len = 512;
    // dilation_heads = 2;
    const int Batch = n_heads, M = seq_len, K = feat_len, W = 32, dilation = 4;
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(n);
    DEFINE_VAR(w);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, 2 * W + 1}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});
    auto subA = makeSubscript(A, {b, m, w});
    // auto subB = makeSubscript(B, {b, m + dilation * (w - W), n});
    auto subB = makeSubscript(B, {b, m - dilation * (w), n});
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M + 1}}, {n, {0, K}}},
                          {{w, {0, 2 * W + 1}}}, subA * subB);
    dbg(range);

    CheckOOBVisitor oobchecker;
    EXPECT_TRUE(oobchecker.checkRangeOp(range));
}