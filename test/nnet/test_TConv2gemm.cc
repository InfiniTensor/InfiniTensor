#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/Visitor/GetTensorsVisitor.h"
#include "nnet/Visitor/Interpreter.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/test.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

TEST(Conv2conv, TConv4x4_NHWC_innerStage_RuleBased) {
    const int N = 1, H = 2, W = 2, C = 256, F = 448;
    const int R = 4, S = 4;
    const int OH = 2 * H, OW = 2 * W;
    DEFINE_VAR(n, c, f, r, s, x1, x2, y1, y2);
    DEFINE_VAR(i2, i4);
    // dilation * (kernel_size - 1) - padding
    int padding = 1 * (R - 1) - 1;
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, F}),
                                  vector<int>{0, padding, padding, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({F, R, S, C}));

    auto subA = makeSubscript(A, {n, x1 + r - 1, y1 + s - 1, f});
    auto subK = makeSubscript(
        K, {f, -2 * r + (-1) * x2 + (R - 1), -2 * s + (-1) * y2 + (S - 1), c});

    auto range = makeRangeOperator(
        {
            {n, {0, N}},
            {c, {0, C}},
            {x1, {0, OH / 2 + 1}},
            {x2, {0, 2}},
            {y1, {0, OW / 2 + 1}},
            {y2, {0, 2}},
        },
        {{f, {0, F}}, {r, {0, R / 2}}, {s, {0, S / 2}}}, subA * subK);
    dbg(range);

    const vector<int> rules{3, 2, 2, 2, 2, 5, 8, 8, 6, 90};
    Formula conv_9x9(range, 0);
    Derivator derivator;
    derivator.ruleBasedDFS(
        conv_9x9, 0, rules,
        {{1, {x1, r}}, {2, {y1, s}}, {3, {x2, i2}}, {4, {y2, i4}}});
    EXPECT_EQ(derivator.getSearchedMaxDepth(), ((int)rules.size()));
    ASSERT_GE(derivator.getNumCandidates(), 1);
    const auto &formula = derivator.getCandidates().front();
    EXPECT_TRUE(CountRoutineVisitor().match(formula.root, 1, 0, 3));
    derivator.print();
}

TEST(Conv2conv, TConv4x4_NHWC_RuleBased) {
    const int N = 1, H = 2, W = 2, C = 256, F = 448;
    const int R = 4, S = 4;
    const int OH = 2 * H, OW = 2 * W;
    DEFINE_VAR(n, c, h, w, f, r, s);
    DEFINE_VAR(x1, x2, y1, y2);
    // dilation * (kernel_size - 1) - padding
    int padding = 1 * (R - 1) - 1;
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, F}),
                                  vector<int>{0, padding, padding, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({F, R, S, C}));

    auto subA = makeSubscript(A, {n, x1 + r - 1, y1 + s - 1, f});
    auto subK = makeSubscript(
        K, {f, -2 * r + (-1) * x2 + (R - 1), -2 * s + (-1) * y2 + (S - 1), c});

    // auto range =
    //     makeRangeOperator({{n, {0, N}}, {c, {0, H}}, {w, {0, W}}, {f, {0,
    //     F}}},
    //                       {{f, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA *
    //                       subK);
    auto range = makeRangeOperator(
        {
            {n, {0, N}},
            {x1, {0, OH / 2 + 1}},
            {x2, {0, 2}},
            {y1, {0, OW / 2 + 1}},
            {y2, {0, 2}},
            {c, {0, C}},
        },
        {{f, {0, F}}, {r, {0, R / 2}}, {s, {0, S / 2}}}, subA * subK);
    auto subOuter = makeSubscript(
        range, {n, (h + 1) / 2, (h + 1) % 2, (w + 1) / 2, (w + 1) % 2, c});
    auto outerRange = makeRangeOperator(
        {
            {n, {0, N}},
            {h, {0, OH}},
            {w, {0, OW}},
            {c, {0, C}},
        },
        {}, subOuter);
    dbg(outerRange);

    // Derivation: this work without padding check in stage merging
    // const vector<int> rules{1, 1, 3, 2, 2, 5, 2, 2, 6, 4, 4, 4, 4, 6};
    // Before Guided DLT seperated from rule2VarMerging
    // const vector<int> rules{1, 1, 3, 2, 2, 5, 2, 2, 6, 6};
    const vector<int> rules{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90};
    Formula conv_9x9(outerRange, 0);
    Derivator derivator;
    // derivator.ruleBasedDFS(conv_9x9, 0, rules,
    //                        {{1, {"x1", "r"}},
    //                         {2, {"y1", "s"}},
    //                         {3, {"x2", "i2"}},
    //                         {4, {"y2", "i4"}}});
    derivator.ruleBasedDFS(conv_9x9, 0, rules);
    EXPECT_EQ(derivator.getSearchedMaxDepth(), ((int)rules.size()));
    ASSERT_GE(derivator.getNumCandidates(), 1);
    const auto &formula = derivator.getCandidates().front();
    EXPECT_TRUE(CountRoutineVisitor().match(formula.root, 1, 0, 3));
    derivator.print();
}

TEST(Conv2conv, TConv4x4_BS16_NHWC_RuleBased) {
    const int N = 16, H = 2, W = 2, C = 256, F = 448;
    const int R = 4, S = 4;
    const int OH = 2 * H, OW = 2 * W;
    DEFINE_VAR(n, c, h, w, f, r, s);
    DEFINE_VAR(x1, x2, y1, y2);
    // dilation * (kernel_size - 1) - padding
    int padding = 1 * (R - 1) - 1;
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, F}),
                                  vector<int>{0, padding, padding, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({F, R, S, C}));

    auto subA = makeSubscript(A, {n, x1 + r - 1, y1 + s - 1, f});
    auto subK = makeSubscript(
        K, {f, -2 * r + (-1) * x2 + (R - 1), -2 * s + (-1) * y2 + (S - 1), c});

    // auto range =
    //     makeRangeOperator({{n, {0, N}}, {c, {0, H}}, {w, {0, W}}, {f, {0,
    //     F}}},
    //                       {{f, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA *
    //                       subK);
    auto range = makeRangeOperator(
        {
            {n, {0, N}},
            {x1, {0, OH / 2 + 1}},
            {x2, {0, 2}},
            {y1, {0, OW / 2 + 1}},
            {y2, {0, 2}},
            {c, {0, C}},
        },
        {{f, {0, F}}, {r, {0, R / 2}}, {s, {0, S / 2}}}, subA * subK);
    auto subOuter = makeSubscript(
        range, {n, (h + 1) / 2, (h + 1) % 2, (w + 1) / 2, (w + 1) % 2, c});
    auto outerRange = makeRangeOperator(
        {
            {n, {0, N}},
            {h, {0, OH}},
            {w, {0, OW}},
            {c, {0, C}},
        },
        {}, subOuter);
    dbg(outerRange);

    // Derivation: this work without padding check in stage merging
    // const vector<int> rules{1, 1, 3, 2, 2, 5, 2, 2, 6, 4, 4, 4, 4, 6};
    // Before Guided DLT seperated from rule2VarMerging
    // const vector<int> rules{1, 1, 3, 2, 2, 5, 2, 2, 6, 6};
    const vector<int> rules{3, 2, 2, 2, 2, 5, 8, 8, 6, 91, 90};
    Formula conv_9x9(outerRange, 0);
    Derivator derivator;
    // derivator.ruleBasedDFS(conv_9x9, 0, rules,
    //                        {{1, {"x1", "r"}},
    //                         {2, {"y1", "s"}},
    //                         {3, {"x2", "i2"}},
    //                         {4, {"y2", "i4"}}});
    derivator.ruleBasedDFS(conv_9x9, 0, rules);
    EXPECT_EQ(derivator.getSearchedMaxDepth(), ((int)rules.size()));
    ASSERT_GE(derivator.getNumCandidates(), 1);
    const auto &formula = derivator.getCandidates().front();
    EXPECT_TRUE(CountRoutineVisitor().match(formula.root, 1, 0, 3));
    derivator.print();
}

// Warn: F is the number of input channels, which is inversed compared with
// normal Conv.
// Our data layout: NHWF -> NHWC, FRSC
// Pytorch data layout: NFHW -> NCHW, FCRS
RangeOp buildTConv4x4_NHWF_FRSC(const int N, const int C, const int H,
                                const int W, const int F, const int R,
                                const int S) {
    assert(R == 4 && S == 4);
    const int OH = 2 * H, OW = 2 * W;
    DEFINE_VAR(n, c, h, w, f, r, s);
    DEFINE_VAR(x1, x2, y1, y2, i2, i4);
    // dilation * (kernel_size - 1) - padding
    int padding = 1 * (R - 1) - 1;
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, F}),
                                  vector<int>{0, padding, padding, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({F, R, S, C}));

    auto subA = makeSubscript(A, {n, x1 + r - 1, y1 + s - 1, f});
    auto subK =
        makeSubscript(K, {f, (R - 2) - 2 * r + x2, (S - 2) - 2 * s + y2, c});
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

// Correct input expression

// Warn: F is the number of input channels, which is inversed compared with
// normal Conv.
// Our data layout: NHWF -> NHWC, RSFC
// Pytorch data layout: NFHW -> NCHW, FCRS
RangeOp buildTConv4x4_NHWF_RSFC(const int N, const int C, const int H,
                                const int W, const int F, const int R,
                                const int S) {
    assert(R == 4 && S == 4);
    const int OH = 2 * H, OW = 2 * W;
    DEFINE_VAR(n, c, h, w, f, r, s);
    DEFINE_VAR(x1, x2, y1, y2, i2, i4);
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

TEST(TConv2gemm, TConv4x4_NHWF_FRSC_correctness_of_input_expr) {
    const int N = 1, H = 2, W = 2, C = 2, F = 3;
    const int R = 4, S = 4;
    RangeOp range0 = buildTConv4x4_NHWF_FRSC(N, C, H, W, F, R, S);

    auto ans0 = Interpreter(range0).interpretAllOutput(range0);
    // Pytorch results
    // torch.conv_transpose2d(X, K, stride=2, padding=1, dilation=1)
    // X, K is NFHW and FCRS
    vector<int> ans1 = {190,  193,  740,  755,  770,  785,  592,  604,
                        992,  1016, 2704, 2770, 2836, 2902, 1832, 1874,
                        1184, 1208, 3232, 3298, 3364, 3430, 2168, 2210,
                        1114, 1135, 2660, 2711, 2762, 2813, 1624, 1654};
    ASSERT_EQ(ans0.size(), ans1.size());
    for (size_t i = 0; i < ans0.size(); ++i)
        EXPECT_EQ(ans0[i], ans1[i]);
}

ssize_t getOffset(vector<ssize_t> index, vector<int> shape) {
    ssize_t ret = index[0];
    for (size_t i = 1; i < index.size(); ++i)
        ret = ret * shape[i] + index[i];
    return ret;
}

TEST(TConv2gemm, TConv4x4_NHWF_RSFC_correctness_of_input_expr) {
    const int N = 1, H = 2, W = 2, C = 2, F = 3;
    const int R = 4, S = 4;
    RangeOp range0 = buildTConv4x4_NHWF_RSFC(N, C, H, W, F, R, S);
    Interpreter::Inputs inputs;

    for (const auto &[name, tensor] : GetTensorsVisitor().get(range0)) {
        auto data = make_ref<vector<int>>(tensor->getSize());
        if (name == "A") {
            for (ssize_t i = 0; i < tensor->getSize(); i++)
                data->operator[](i) = i;
        } else if (name == "K") {
            for (ssize_t r = 0; r < R; r++)
                for (ssize_t s = 0; s < S; s++)
                    for (ssize_t f = 0; f < F; f++)
                        for (ssize_t c = 0; c < C; c++) {
                            ssize_t index =
                                getOffset({r, s, f, c}, {R, S, F, C});
                            ssize_t num = getOffset({f, r, s, c}, {F, R, S, C});
                            data->operator[](index) = num;
                        }

        } else
            assert(0);
        inputs.emplace(name, data);
    }

    auto ans0 = Interpreter(inputs).interpretAllOutput(range0);
    // Pytorch results
    // torch.conv_transpose2d(X, K, stride=2, padding=1, dilation=1)
    // X, K is NFHW and FCRS
    vector<int> ans1 = {190,  193,  740,  755,  770,  785,  592,  604,
                        992,  1016, 2704, 2770, 2836, 2902, 1832, 1874,
                        1184, 1208, 3232, 3298, 3364, 3430, 2168, 2210,
                        1114, 1135, 2660, 2711, 2762, 2813, 1624, 1654};
    ASSERT_EQ(ans0.size(), ans1.size());
    for (size_t i = 0; i < ans0.size(); ++i)
        EXPECT_EQ(ans0[i], ans1[i]);
}

// TODO: Test after passing RSFC
// TEST(TConv2gemm, TConv4x4_NHWF_FRSC_search) {
//     const int N = 1, H = 2, W = 2, C = 256, F = 448;
//     const int R = 4, S = 4;
//     RangeOp range = buildTConv4x4_NHWF_FRSC(N, C, H, W, F, R, S);

//     const vector<int> rules{3, 2, 2, 2, 2, 5, 8, 8, 6, 90};
//     Formula conv_9x9(range, 0);
//     Derivator derivator;
//     // derivator.ruleBasedDFS(
//     //     conv_9x9, 0, rules,
//     //     {{1, {x1, r}}, {2, {y1, s}}, {3, {x2, i2}}, {4, {y2, i4}}});
//     derivator.dfs(conv_9x9, 0);
//     // EXPECT_EQ(derivator.getSearchedMaxDepth(), ((int)rules.size()));
//     ASSERT_GE(derivator.getNumCandidates(), 1);
//     const auto &formula = derivator.getCandidates().front();
//     EXPECT_TRUE(CountRoutineVisitor().match(formula.root, 1, 0, 3));
//     derivator.print();
// }

TEST(TConv2gemm, TConv4x4_NHWF_RSFC_search) {
    const int N = 16, H = 2, W = 2, C = 256, F = 448;
    const int R = 4, S = 4;
    RangeOp range = buildTConv4x4_NHWF_RSFC(N, C, H, W, F, R, S);

    Formula conv_9x9(range, 0);
    Derivator derivator;

    bool isRuleBased = false;
    if (isRuleBased) {
        const vector<int> rules{3, 2, 2, 2, 2, 5};
        derivator.setDumpFirstSuccess("TConv4x4_NHWF_RSFC_");
        derivator.ruleBasedDFS(conv_9x9, 0, rules, {}, true);
    } else
        derivator.search(conv_9x9, 0);

    ASSERT_GE(derivator.getNumCandidates(), 1);
    derivator.print();
    // for (const auto &f : derivator.getCandidates()) {
    //     dbg(CountRoutineVisitor().count(f.root));
    // }
    int nMatches = matchExprResult(
        derivator,
        "../test/nnet/log/TConv4x4_NHWF_RSFC/TConv4x4_NHWF_RSFC_18.expr");
    EXPECT_GE(nMatches, 1);
    derivator.printStatistics();
}

TEST(TConv2gemm, TConv4x4_NHWF_FRSC_CheckDerivationCorrectness_log) {
    const string fnPrefix =
        "../test/nnet/log/TConv4x4_NHWF_RSFC/TConv4x4_NHWF_RSFC_";
    EXPECT_TRUE(checkExprLogSame(fnPrefix, 0, 11));
}

// TODO: correct ConvTransPattern
TEST(Conv2conv, InfoGAN_ConvTranspose_3_OOB_Test) {
    // ConvTranspose_3 in InfoGAN
    const int n = 1, c = 256, h = 2, w = 2, f = 448, r = 4, s = 4;
    int padding = 1 * (r - 1) - 1;
    const auto A = nnet::makeTensor("A", {n, h, w, f},
                                    std::vector<int>{0, padding, padding, 0});
    const auto K = nnet::makeTensor("K", {f, c, r, s});
    auto expr = ConvTransPattern::getExpr(A, K, n, c, h, w, f, r, s);
    dbg(expr);
    Derivator derivator;
    derivator.checkOOB(as<RangeOpNode>(expr));
}
