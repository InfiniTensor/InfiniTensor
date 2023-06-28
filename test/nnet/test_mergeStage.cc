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

    auto subA = mSub(A, {b, m, k});
    auto innerRange = mL({{b, {0, Batch}}, {m, {0, M}}, {k, {0, K}}}, {}, subA);
    auto relu = make_ref<FuncNode>(subA, FuncType::Relu);
    auto range = mL({{b, {0, Batch}}, {m, {0, M}}, {w, {0, 2 * W + 1}}},
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

    auto subA = mSub(B, {b, k, m});
    auto range = mL({{b, {0, Batch}}, {m, {0, M}}}, {{k, {0, K}}}, subA);
    auto innerRange =
        mL({{b, {0, Batch}}, {k, {0, K}}, {m, {0, M}}}, {}, mSub(A, {b, m, k}));
    auto merged = MergeMemboundMutator({innerRange, range}).merge();
    RangeOp ans =
        mL({{b, {0, Batch}}, {m, {0, M}}}, {{k, {0, K}}}, mSub(A, {b, m, k}));
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

    auto subA = mSub(K, {f, i / 1024, (i / 256) % 4, i % 256});
    auto range = mL({{i, {0, I}}, {f, {0, F}}}, {}, subA);
    auto outerRange = mL({{f, {0, F}}, {i, {0, I}}}, {}, mSub(range, {i, f}));
    auto merged = MergeMemboundMutator({outerRange}).merge();

    // Compare the result with answer
    RangeOp ans = mL({{f, {0, F}}, {i, {0, I}}}, {},
                     mSub(K, {f, i / 1024, (i / 256) % 4, i % 256}));
    EXPECT_EQ(HashVisitor().getHash(merged), HashVisitor().getHash(ans));
}

TEST(FuseMembound, mergeReductionBiasRelu) {
    DEFINE_VAR(f, i);
    const int F = 4, H = 16;
    auto A = mT("A", vector<int>({F, H}));
    auto B = mT("B", vector<int>({F, H}));
    auto AB = mT("AB", vector<int>({F, H}));
    auto C = mT("Bias", vector<int>({F}));
    auto l0 = // Reduction
        mL({{f, {0, F}}, {i, {0, H}}}, {}, mSub(A, {f, i}) * mSub(B, {f, i}));
    auto l1 = // Bias
        mL({{f, {0, F}}, {i, {0, H}}}, {}, mSub(l0, {f, i}) + mSub(C, {f}));
    // Relu
    auto l2 = mL({{f, {0, F}}, {i, {0, H}}}, {},
                 make_ref<FuncNode>(mSub(AB, {f, i}), FuncType::Relu));
    dbg(l1, l2);

    auto merged = MergeMemboundMutator({l1, l2}).merge();
    dbg(merged);

    // TODO:
    // 1. 用NMutator::constructGraphFromExpression，把以上merged表达式变为算子，
    // 跑通TVM codegen。现在端到端运行模型用的是test/nnet/run_models_nnet.py的
    // model_e2e_exp()，可将以上函数作为整体暴露给python。
    // 2. 在NMutator::runSingleOp中，处理带有bias、relu的conv，使得输入一个conv
    // operator，能得到Gemm加一个Membound算子(Reduce+Bias+Relu)。现在的代码中，
    // NMutator::opToExpression返回的表达式会直接忽略bias和relu。
}
