#include "core/graph.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/Visitor/MergeMemboundMutator.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(FuseMembound, Relu) {
    const int n_heads = 8, seq_len = 10000, feat_len = 512;
    // dilation_heads = 2;
    const int Batch = n_heads, M = seq_len, K = feat_len, W = 32;
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});

    auto subA = makeSubscript(A, {b, m, k});
    auto innerRange = makeRangeOperator(
        {{b, {0, Batch}}, {m, {0, M}}, {k, {0, K}}}, {}, subA);
    auto relu = make_ref<FuncNode>(subA, FuncType::Relu);
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {w, {0, 2 * W + 1}}},
                          {{k, {0, K}}}, relu);
    dbg(range);
    dbg(MergeMemboundMutator({range, innerRange}).merge());
    cout << MergeMemboundMutator({range, innerRange}).merge()->toReadable()
         << endl;
}

TEST(FuseMembound, MemMemFusion) {
    const int n_heads = 8, seq_len = 100, feat_len = 100;
    // dilation_heads = 2;
    const int Batch = n_heads, M = seq_len, K = feat_len;
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, K, M}),
                                  vector<int>{0, 0, 0});

    auto subA = makeSubscript(B, {b, k, m});
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}}, {{k, {0, K}}}, subA);
    auto innerRange =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {k, {0, K}}}, {},
                          makeSubscript(A, {b, k, m}));
    dbg(range, innerRange);
    auto merged = MergeMemboundMutator({range, innerRange}).merge();
    dbg(merged);
    RangeOp ans = makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}},
                                    {{k, {0, K}}}, makeSubscript(A, {b, m, k}));
    EXPECT_EQ(HashVisitor().getHash(merged), HashVisitor().getHash(ans));
}