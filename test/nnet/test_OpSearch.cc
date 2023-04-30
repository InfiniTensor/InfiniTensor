#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/test.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

// clang-format off
/* Evaluation bash script
# Maxdepth
for i in $(seq 1 12); do
    echo $i
    NNET_UseHash=1 NNET_MaxDepth=$i ./test_OpSearch &> out.searchDepthTest.$i.txt done

# Enable/disable hash
NNET_UseHash=0 NNET_MaxDepth=8 ./test_OpSearch &> out.searchDepthTest.$i.txt
NNET_UseHash=1 NNET_MaxDepth=8 ./test_OpSearch &> out.searchDepthTest.$i.txt

# Number of derivation steps
for i in Conv3x3 ConvTranspose Conv5x5 G2BMM; do
    NNET_PrintAndExit=1 NNET_UseHash=1 NNET_MaxDepth=7 ./test_OpSearch --gtest_filter="*$i" > out.steps.$i.txt done
*/
// clang-format on

class OpSearch : public ::testing::Test {
  protected:
    const int maxDepth = getMaxDepth();
    const int useHash = getUseHash();
    const bool printAndExit = getPrintAndExit();
    // const int maxDepth = 8;
    // const int useHash = true;
    const Derivator::LogMode mode = getPrintAndExit()
                                        ? Derivator::LogMode::DumpFristCandiate
                                        : Derivator::LogMode::NoLog;
    const Derivator::PassMode passMode = Derivator::PassMode::Full;
    const bool isRuleBased = getPrintAndExit();

    void SetUp() override {
        if (maxDepth < 0 || useHash < 0) {
            GTEST_SKIP() << "Skipping OpSearch since NNET_MaxDepth or "
                            "NNET_UseHash are not specifed.\n";
        }
    }

  private:
    static int getMaxDepth() {
        if (auto s = getenv("NNET_MaxDepth"))
            return atoi(s);
        return -1;
    }

    static bool getUseHash() {
        if (auto s = getenv("NNET_UseHash"))
            return atoi(s);
        return -1;
    }

    static bool getPrintAndExit() {
        if (auto s = getenv("NNET_PrintAndExit"))
            return atoi(s);
        return 0;
    }
};

// TEST_F(OpSearch, Conv2gemm_NCHW_FCRS_search) {
TEST_F(OpSearch, Conv3x3) {
    // A[n,h+r,w+s,c]*K[r,s,f,c]
    int N = 1, H = 7, W = 7, C = 512, F = 512;
    int R = 3, S = 3;
    auto n = make_ref<VarNode>("n");
    auto c = make_ref<VarNode>("c");
    auto h = make_ref<VarNode>("h");
    auto w = make_ref<VarNode>("w");
    auto f = make_ref<VarNode>("f");
    auto r = make_ref<VarNode>("r");
    auto s = make_ref<VarNode>("s");
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, C}),
                                  vector<int>{0, R / 2, S / 2, 0});
    // auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, h + r - R / 2, w + s - S / 2, c});
    auto subK = makeSubscript(K, {r, s, f, c});

    auto range =
        makeRangeOperator({{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subK);

    // Derivation
    Formula conv_3x3_nhwc_rsfc(range, 0);
    Derivator derivator(maxDepth, useHash, mode, passMode, printAndExit);

    if (isRuleBased) {
        // Rule-based derivation
        const vector<int> rules = {3, 2, 2, 5, 8, 8, 6, 90};
        derivator.setDumpFirstSuccess("Conv2gemm_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_3x3_nhwc_rsfc, 0, rules);
    } else {
        derivator.search(conv_3x3_nhwc_rsfc, 0);
    }

    EXPECT_GE(derivator.getNumCandidates(), 1);
    int nMatches = matchExprResult(
        derivator, "../test/nnet/log/conv2gemm/Conv2gemm_NCHW_FCRS_11.expr");
    EXPECT_GE(nMatches, 1);
    // derivator.print();
    derivator.printStatistics();
}

// Warn: F is the number of input channels, which is inversed compared with
// normal Conv.
// Our data layout: NHWF -> NHWC, RSFC
// Pytorch data layout: NFHW -> NCHW, FCRS
RangeOp buildTConv4x4_NHWF_RSFC(const int N, const int C, const int H,
                                const int W, const int F, const int R,
                                const int S) {
    assert(R == 4 && S == 4);
    const int OH = 2 * H, OW = 2 * W;
    DEFINE_VAR(n);
    DEFINE_VAR(c);
    DEFINE_VAR(f);
    DEFINE_VAR(r);
    DEFINE_VAR(s);
    DEFINE_VAR(x1);
    DEFINE_VAR(x2);
    DEFINE_VAR(y1);
    DEFINE_VAR(y2);
    DEFINE_VAR(i2);
    DEFINE_VAR(i4);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    // dilation * (kernel_size - 1) - padding
    int padding = 1 * (R - 1) - 1;
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, F}),
                                  vector<int>{0, padding, padding, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, x1 + r - 1, y1 + s - 1, f});
    auto subK =
        makeSubscript(K, {(R - 2) - 2 * r + x2, (S - 2) - 2 * s + y2, f, c});
    // x1=(h+1)//2, x2=(h+1)%2, y1=(w+1)//2

    auto range1 = makeRangeOperator(
        {
            {n, {0, N}},
            {c, {0, C}},
            {x1, {0, OH / 2 + 1}},
            {x2, {0, 2}},
            {y1, {0, OW / 2 + 1}},
            {y2, {0, 2}},
        },
        {{f, {0, F}}, {r, {0, R / 2}}, {s, {0, S / 2}}}, subA * subK);
    dbg(range1);
    auto sub0 = makeSubscript(
        range1, {n, c, (h + 1) / 2, (h + 1) % 2, (w + 1) / 2, (w + 1) % 2});
    auto range0 = makeRangeOperator(
        {{n, {0, N}}, {h, {0, OH}}, {w, {0, OW}}, {c, {0, C}}}, {}, sub0);
    return range0;
}

// TEST_F(OpSearch, TConv2gemm_TConv4x4_NHWF_RSFC_search) {
TEST_F(OpSearch, ConvTranspose) {
    const int N = 16, H = 2, W = 2, C = 256, F = 448;
    const int R = 4, S = 4;
    RangeOp range = buildTConv4x4_NHWF_RSFC(N, C, H, W, F, R, S);

    Formula conv_9x9(range, 0);
    Derivator derivator(maxDepth, useHash, mode, passMode, printAndExit);

    if (isRuleBased) {
        const vector<int> rules{3, 2, 2, 2, 2, 5};
        derivator.setDumpFirstSuccess("TConv4x4_NHWF_RSFC_");
        derivator.ruleBasedDFS(conv_9x9, 0, rules, {}, true);
    } else
        derivator.search(conv_9x9, 0);

    EXPECT_GE(derivator.getNumCandidates(), 1);
    int nMatches = matchExprResult(
        derivator,
        "../test/nnet/log/TConv4x4_NHWF_RSFC/TConv4x4_NHWF_RSFC_18.expr");
    EXPECT_GE(nMatches, 1);
    derivator.printStatistics();
}

// TEST_F(OpSearch, Conv2conv_5x5_RuleBased_NCHW_FCRS) {
TEST_F(OpSearch, Conv5x5) {
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
    Derivator derivator(maxDepth, useHash, mode, passMode, printAndExit);

    if (isRuleBased) {
        const vector<int> rules{9, 1, 1, 3, 2, 2, 5, 8, 8, 6, 6};
        derivator.setDumpFirstSuccess("Conv2conv_5x5_NCHW_FCRS_");
        derivator.ruleBasedDFS(conv_9x9, 0, rules, {}, true);
    } else
        derivator.search(conv_9x9, 0);

    int nMatches = matchExprResult(
        derivator,
        "../test/nnet/log/conv2conv/Conv2conv_5x5_NCHW_FCRS_15.expr");
    // derivator.print();
    derivator.printStatistics();
    EXPECT_GE(nMatches, 1);
}

// TEST_F(OpSearch, G2BMM_RuleBased) {
TEST_F(OpSearch, G2BMM) {
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
    Derivator derivator(maxDepth, useHash, mode, passMode, printAndExit);

    if (isRuleBased) {
        const vector<int> rules{1, 7, 7, 2, 8, 6, 6};
        derivator.setDumpFirstSuccess("G2BMM_");
        derivator.ruleBasedDFS(dialted_g2bmm, 0, rules);
    } else {
        derivator.search(dialted_g2bmm, 0);
    }

    EXPECT_GE(derivator.getNumCandidates(), 1);
    int nMatches =
        matchExprResult(derivator, "../test/nnet/log/g2bmm/G2BMM_9.expr");
    EXPECT_GE(nMatches, 1);
    // derivator.print();
    derivator.printStatistics();
}
