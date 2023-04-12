#include "core/graph.h"
#include "core/operator.h"
#include "core/tensor.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/Visitor/MergeMemboundMutator.h"
#include "nnet/expr.h"
#include "nnet/test.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

TEST(FuseMembound, Relu) {
    const int n_heads = 8, seq_len = 10000, feat_len = 512;
    // dilation_heads = 2;
    const int Batch = n_heads, M = seq_len, K = feat_len, W = 32;
    DEFINE_VAR(b, m, w, k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});

    auto subA = makeSubscript(A, {b, m, k});
    auto innerRange = makeRangeOperator(
        {{b, {0, Batch}}, {m, {0, M}}, {k, {0, K}}}, {}, subA);
    auto relu = make_ref<FuncNode>(subA, FuncType::Relu);
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {w, {0, 2 * W + 1}}},
                          {{k, {0, K}}}, relu);
    cout << MergeMemboundMutator({innerRange, range}).merge()->toReadable()
         << endl;
}

TEST(FuseMembound, MemMemFusion) {
    const int n_heads = 8, seq_len = 100, feat_len = 100;
    // dilation_heads = 2;
    const int Batch = n_heads, M = seq_len, K = feat_len;
    DEFINE_VAR(b, m, w, k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});
    auto B = make_ref<TensorNode>("B", vector<int>({Batch, K, M}),
                                  vector<int>{0, 0, 0});

    auto subA = makeSubscript(B, {b, k, m});
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}}, {{k, {0, K}}}, subA);
    auto innerRange =
        makeRangeOperator({{b, {0, Batch}}, {k, {0, K}}, {m, {0, M}}}, {},
                          makeSubscript(A, {b, m, k}));
    auto merged = MergeMemboundMutator({innerRange, range}).merge();
    RangeOp ans = makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}},
                                    {{k, {0, K}}}, makeSubscript(A, {b, m, k}));
    EXPECT_EQ(HashVisitor().getHash(merged), HashVisitor().getHash(ans));
}

TEST(FuseMembound, mergeNestedStagesInRangeOp) {
    // Case in ConvTranspose to Matmul
    // L<f:0:448><i39:0:4096>Sum  ...  [i39,f]
    //   {L<i39:0:4096><f:0:448>Sum  ...  [f,(i39 / 1024),((i39 / 256) % 4),(i39
    //   % 256)] {K}}
    DEFINE_VAR(f, i);
    const int I = 4096, F = 448;
    auto K = make_ref<TensorNode>("K", vector<int>({448, 4, 4, 256}));

    auto subA = makeSubscript(K, {f, i / 1024, (i / 256) % 4, i % 256});
    auto range = makeRangeOperator({{i, {0, I}}, {f, {0, F}}}, {}, subA);
    auto outerRange = makeRangeOperator({{f, {0, F}}, {i, {0, I}}}, {},
                                        makeSubscript(range, {i, f}));
    auto merged = MergeMemboundMutator({outerRange}).merge();

    // Compare the result with answer
    RangeOp ans = makeRangeOperator(
        {{f, {0, F}}, {i, {0, I}}}, {},
        makeSubscript(K, {f, i / 1024, (i / 256) % 4, i % 256}));
    EXPECT_EQ(HashVisitor().getHash(merged), HashVisitor().getHash(ans));
}
