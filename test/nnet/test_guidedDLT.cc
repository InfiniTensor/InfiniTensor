#include "nnet/Pass/Rule8GuidedDLT.h"
#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/iterator_table.h"
#include "nnet/permutation.h"
#include "nnet/test.h"
using namespace nnet;
using namespace std;

TEST(GuidedDLT, Permuation) {
    DEFINE_VAR(_Conv_c, _Conv_r, _Conv_s, _Conv_h, _Conv_n, _Conv_w);
    DEFINE_VAR(c, i14, i4, i17, i22, n);
    PermutationGenerator permutator{
        {{_Conv_c, _Conv_r, _Conv_s}, {_Conv_h, _Conv_n, _Conv_w}},
        {{c, i14, i4}, {i17, i22, n}}};
    int cnt = 0;
    do {
        cnt++;
        dbg(permutator.get());
    } while (permutator.next());
    EXPECT_EQ(cnt, 6 * 6);
}

TEST(GuidedDLT, dimFusion_ConvToGemm_1Tensor) {
    int N = 8, K = 16;
    DEFINE_VAR(r, s, n, t1, t2, f, c);
    auto A = make_ref<TensorNode>("A", vector<int>({N, N, N, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {n, t1, t2, c});
    auto subB = makeSubscript(B, {r, c});
    auto range = makeRangeOperator(
        {{n, {0, N}}, {t1, {0, N}}, {t2, {0, N}}, {r, {0, N}}}, {{c, {0, K}}},
        subA * subB);
    // Derivation
    Derivator derivator(2);
    {
        dbg(range);
        Formula matmul(range, 0);
        Derivator derivator(3);
        Rule8GuidedDLT pass(derivator);
        auto ret = pass.guidedDLT(matmul, 1, matmul.root, true);
        ASSERT_GE(ret.size(), 1u);
        dbg(ret);
        EXPECT_EQ(ret.size(), 1u);
        auto rangeOp = as<RangeOpNode>(ret[0]);
        ASSERT_TRUE(rangeOp != nullptr);
        EXPECT_EQ(rangeOp->getLoopVarRanges().size(), 4u);
        EXPECT_EQ(rangeOp->getSumVarRanges().size(), 0u);
        dbg(rangeOp, rangeOp->getSummand());
        auto sub = as<SubscriptNode>(rangeOp->getSummand());
        ASSERT_TRUE(sub != nullptr);
        auto inner = as<RangeOpNode>(sub->getObject());
        ASSERT_TRUE(inner != nullptr);
        EXPECT_EQ(inner->getSumVarRanges().size(), 1u);
        EXPECT_EQ(inner->getLoopVarRanges().size(), 2u);
    }
}

TEST(GuidedDLT, dimFusion_ConvToGemm_1step) {
    int N = 8, K = 16;

    DEFINE_VAR(r, s, n, t1, t2, f, c);
    auto A = make_ref<TensorNode>("A", vector<int>({N, N, N, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, N, N, K}));

    auto subA = makeSubscript(A, {n, t1, t2, c});
    auto subB = makeSubscript(B, {r, s, f, c});
    auto range = makeRangeOperator({{r, {0, N}},
                                    {s, {0, N}},
                                    {n, {0, N}},
                                    {t1, {0, N}},
                                    {t2, {0, N}},
                                    {f, {0, N}}},
                                   {{c, {0, K}}}, subA * subB);
    // Derivation
    {
        Formula matmul(range, 0);
        Derivator derivator(1);
        Rule8GuidedDLT pass(derivator);
        auto ret = pass.guidedDLT(matmul, 1, matmul.root, true);
        ASSERT_GE(ret.size(), 1u);
        dbg(ret);
        for (const auto &cur : ret) {
            auto rangeOp = as<RangeOpNode>(cur);
            ASSERT_TRUE(rangeOp != nullptr);
            EXPECT_EQ(rangeOp->getLoopVarRanges().size(), 6u);
            EXPECT_EQ(rangeOp->getSumVarRanges().size(), 0u);
            dbg(rangeOp, rangeOp->getSummand());
            auto sub = as<SubscriptNode>(rangeOp->getSummand());
            ASSERT_TRUE(sub != nullptr);
            auto inner = as<RangeOpNode>(sub->getObject());
            ASSERT_TRUE(inner != nullptr);
            EXPECT_EQ(inner->getSumVarRanges().size(), 1u);
            EXPECT_EQ(inner->getLoopVarRanges().size(), 4u);
        }
    }
}

TEST(GuidedDLT, dimFusion_ConvToGemm_real_2tensors) {
    int N = 8, K = 16;

    DEFINE_VAR(r, s, n, t1, t2, f, c);
    auto A = make_ref<TensorNode>("A", vector<int>({N, N, N, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, N, N, K}));

    auto subA = makeSubscript(A, {n, t1, t2, c});
    auto subB = makeSubscript(B, {r, s, f, c});
    auto range = makeRangeOperator({{r, {0, N}},
                                    {s, {0, N}},
                                    {n, {0, N}},
                                    {t1, {0, N}},
                                    {t2, {0, N}},
                                    {f, {0, N}}},
                                   {{c, {0, K}}}, subA * subB);
    // Derivation
    {
        Formula matmul(range, 0);
        Derivator derivator(2);
        const vector<int> rules = {8, 8};
        derivator.ruleBasedDFS(matmul, 0, rules);
        EXPECT_EQ(derivator.getSearchedMaxDepth(), 2);
    }
}

TEST(GuidedDLT, Conv2Conv_KernelDLT) {
    int N = 8, H = 224, W = 224, C = 16, F = 32;
    int R = 9, S = 9;
    DEFINE_VAR(i19, i20, j15, j16, j14, j4, n, f, c);
    // auto A =
    //     make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
    //     vector<int>{0, 0, 1, 1});
    auto A = makeTensor("A", {N, C, H, W}, {0, 0, 3, 3});
    auto B = make_ref<TensorNode>("W", vector<int>({F, C, R, S}));
    // cur =
    // L<i19:-1:226><i20:-1:2><i15:-1:226><i16:-1:2><n:0:8><f:0:32><pad=2,0,2,0,0,0,>Sum<i14:-1:2><i4:-1:2><c:0:16>
    //     {({A<pad=0,0,4,4>}[n, c, (i15 + i4), (i14 + i19)] * {K}[f, c, ((3 *
    //     i16) + i4), (i14 + (3 * i20))])} (std::shared_ptr<nnet::RangeOpNode>)

    auto subA = makeSubscript(A, {n, c, (j15 + j4 - 1), (j14 - 1 + i19)});
    auto subB = makeSubscript(B, {f, c, ((3 * j16) + j4), (j14 + (3 * i20))});
    auto range = makeRangeOperator({{i19, {-1, 226}},
                                    {i20, {0, 3}},
                                    {j15, {-1, 226}},
                                    {j16, {0, 3}},
                                    {n, {0, 8}},
                                    {f, {0, 32}}},
                                   {{j14, {0, 3}}, {j4, {0, 3}}, {c, {0, 16}}},
                                   subA * subB);
    // Derivation
    {
        Formula conv(range, 0);
        Derivator derivator(2);
        derivator.setSearchState(1);
        Rule8GuidedDLT pass(derivator);
        auto ret = pass.guidedDLT(conv, 1, conv.root, true);
        ASSERT_GE(ret.size(), 1u);
        EXPECT_EQ(ret.size(), 1u);
        auto rangeOp = as<RangeOpNode>(ret[0]);
        ASSERT_TRUE(rangeOp != nullptr);
        EXPECT_EQ(rangeOp->getLoopVarRanges().size(), 6u);
        EXPECT_EQ(rangeOp->getSumVarRanges().size(), 0u);
        dbg(rangeOp, rangeOp->getSummand());
        auto sub = as<SubscriptNode>(rangeOp->getSummand());
        ASSERT_TRUE(sub != nullptr);
        auto inner = as<RangeOpNode>(sub->getObject());
        ASSERT_TRUE(inner != nullptr);
        EXPECT_EQ(inner->getSumVarRanges().size(), 3u);
        EXPECT_EQ(inner->getLoopVarRanges().size(), 4u);
    }
}

// TEST(GuidedDLT, Conv2Conv_outputDLT) {
//     int N = 8, H = 224, W = 224, C = 16, F = 32;
//     int R = 9, S = 9;
//     DEFINE_VAR(j101);
//     DEFINE_VAR(j55);
//     DEFINE_VAR(j79);
//     DEFINE_VAR(j14);
//     DEFINE_VAR(j4);
//     DEFINE_VAR(n);
//     DEFINE_VAR(c);
//     auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}));
//     auto B = make_ref<TensorNode>("W", vector<int>({F, C, R, S}));
//     //
//     {L<i101:0:288><i79:-3:227><i55:-3:227><n:0:8>Sum<i14:-1:2><i4:-1:2><c:0:16>
//     // {({A<pad=0,0,4,4>}[n, c, (i4 + i55), (i14 + i79)] * {T1}[i101, c, i4,
//     // i14])}}}}}
//     auto subA = makeSubscript(A, {n, c, (j4 + j55), (j14 + j79)});
//     auto subB = makeSubscript(B, {j101, c, j4, j14});
//     auto range = makeRangeOperator(
//         {{j101, {0, 288}}, {j79, {-3, 227}}, {j55, {-3, 227}}, {n, {0, 8}}},
//         {{j14, {-1, 2}}, {j4, {-1, 2}}, {c, {0, 16}}}, subA * subB);
//     // Derivation
//     {
//         Formula conv(range, 0);
//         Derivator derivator(2);
//         auto ret = derivator.guidedDLT(conv, 1, conv.root, true);
//         dbg(ret);
//         ASSERT_GE(ret.size(), 1);
//         EXPECT_EQ(ret.size(), 1);
//         auto rangeOp = as<RangeOpNode>(ret[0]);
//         ASSERT_TRUE(rangeOp != nullptr);
//         EXPECT_EQ(rangeOp->getLoopVarRanges().size(), 4);
//         EXPECT_EQ(rangeOp->getSumVarRanges().size(), 0);
//         dbg(rangeOp, rangeOp->getSummand());
//         auto sub = as<SubscriptNode>(rangeOp->getSummand());
//         ASSERT_TRUE(sub != nullptr);
//         auto inner = as<RangeOpNode>(sub->getObject());
//         ASSERT_TRUE(inner != nullptr);
//         EXPECT_EQ(inner->getSumVarRanges().size(), 3);
//         ASSERT_EQ(inner->getLoopVarRanges().size(), 4);
//         const auto expectedOrder = vector{n, j101, j55, j79};
//         for (int i = 0; i < 4; ++i) {
//             EXPECT_EQ(inner->getLoopVar(i)->getName(),
//                       expectedOrder[i]->getName());
//         }
//     }
// }

TEST(GuidedDLT, dimFusion_ConvToGemm_2Tensor_ruleBased) {
    int N = 8, K = 16;

    DEFINE_VAR(r, s, n, t1, t2, f, c);
    auto A = make_ref<TensorNode>("A", vector<int>({N, N, N, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, N, N, K}));

    auto subA = makeSubscript(A, {n, t1, t2, c});
    auto subB = makeSubscript(B, {r, s, f, c});
    auto range = makeRangeOperator({{r, {0, N}},
                                    {s, {0, N}},
                                    {n, {0, N}},
                                    {t1, {0, N}},
                                    {t2, {0, N}},
                                    {f, {0, N}}},
                                   {{c, {0, K}}}, subA * subB);
    // Derivation
    Formula matmul(range, 0);
    {
        Derivator derivator(5);
        // derivator.ruleBasedDFS(matmul, 0, {2, 2, 2, 6, 4, 4, 6});
        // derivator.ruleBasedDFS(matmul, 0, {2, 2, 2, 6, 6});
        derivator.ruleBasedDFS(matmul, 0, {8, 8, 6, 6});
        dbg(derivator.getNumCandidates());
        EXPECT_GT(derivator.getNumCandidates(), 0);
        bool simplestMatched = false;
        for (const auto &formula : derivator.getCandidates()) {
            auto routineCnts = CountRoutineVisitor().count(formula.root);
            if (routineCnts[routineTypeToId(
                    RoutineType::ElementWiseNodeType)] == 3 &&
                routineCnts[routineTypeToId(RoutineType::MatmulNodeType)] == 1)
                simplestMatched = true;
        }
        EXPECT_TRUE(simplestMatched);
    }
}

TEST(GuidedDLT, dimFusion_ConvToGemm_2Tensor_dfs) {
    int N = 8, K = 16;

    DEFINE_VAR(r, s, n, t1, t2, f, c);
    auto A = make_ref<TensorNode>("A", vector<int>({N, N, N, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, N, N, K}));

    auto subA = makeSubscript(A, {n, t1, t2, c});
    auto subB = makeSubscript(B, {r, s, f, c});
    auto range = makeRangeOperator({{r, {0, N}},
                                    {s, {0, N}},
                                    {n, {0, N}},
                                    {t1, {0, N}},
                                    {t2, {0, N}},
                                    {f, {0, N}}},
                                   {{c, {0, K}}}, subA * subB);
    // Derivation
    Formula matmul(range, 0);
    {
        Derivator derivator(0);
        derivator.search(matmul, 0);
        dbg(derivator.getNumCandidates());
        EXPECT_GT(derivator.getNumCandidates(), 0);
        bool simplestMatched = false;
        for (const auto &formula : derivator.getCandidates()) {
            auto routineCnts = CountRoutineVisitor().count(formula.root);
            // dbg("&&&&&&&&&&&&&&&&&&&&&", formula.bfsDepth, formula.root,
            //     routineCnts);
            // dbg(FullPrinterVisitor().print(formula.root));
            if (routineCnts[routineTypeToId(
                    RoutineType::ElementWiseNodeType)] == 3 &&
                routineCnts[routineTypeToId(RoutineType::MatmulNodeType)] == 1)
                simplestMatched = true;
        }
        EXPECT_TRUE(simplestMatched);
    }
}

//     {L<i21:0:576><i19:2:228><i15:2:228><n:0:1>Sum<i14:0:3><i4:0:3><c:0:1>
//     {({A<pad=0,0,4,4>}[n, c, ((i15 + i4) + -4), ((i14 + i19) + -4)] *
//     {T1}[i21, c, i4, i14])}}}
// ==> A : Input Tensor shape=[1,1,224,224] pad=[0,0,4,4]
// ==> T1 : EleWise{K, }
// L<i21:0:576><c:0:1><i4:0:3><i14:0:3>Sum  ...  [(i21 / 9),c,((3 * ((i21 / 3) %
// 3)) + i4),(i14 + (3 * (i21 % 3)))]
//     {K}
// ==> K : Input Tensor shape=[64,1,9,9] pad=[0,0,0,0]

// Disabled since forget the answer
TEST(GuidedDLT, DISABLED_match_ConvToConv_conv) {
    DEFINE_VAR(r, s, n, i22, i4, i14, i17, i24, f, c);
    auto A = makeTensor("A", {1, 1, 224, 224}, {0, 0, 4, 4});
    auto B = make_ref<TensorNode>("B", vector<int>({576, 1, 3, 3}));

    auto subA = makeSubscript(A, {n, c, ((i22 + i4) + -4), ((i14 + i17) + -4)});
    auto subB = makeSubscript(B, {i24, c, i4, i14});
    auto range = makeRangeOperator(
        {{i24, {0, 576}}, {i22, {2, 228}}, {i17, {2, 228}}, {n, {0, 1}}},
        {{i14, {0, 3}}, {i4, {0, 3}}, {c, {0, 1}}}, subA * subB);
    dbg(range);
    // Derivation
    {
        Formula conv(range, 0);
        Derivator derivator(2);
        Rule8GuidedDLT pass(derivator);
        auto ret = pass.guidedDLT(conv, 1, conv.root, true);
        dbg(ret);
        ASSERT_EQ(ret.size(), 1u);
        // ASSERT_GE(ret.size(), 1);
        // EXPECT_EQ(ret.size(), 1);
        // auto rangeOp = as<RangeOpNode>(ret[0]);
        // ASSERT_TRUE(rangeOp != nullptr);
        // EXPECT_EQ(rangeOp->getLoopVarRanges().size(), 4);
        // EXPECT_EQ(rangeOp->getSumVarRanges().size(), 0);
        // dbg(rangeOp, rangeOp->getSummand());
        // auto sub = as<SubscriptNode>(rangeOp->getSummand());
        // ASSERT_TRUE(sub != nullptr);
        // auto inner = as<RangeOpNode>(sub->getObject());
        // ASSERT_TRUE(inner != nullptr);
        // EXPECT_EQ(inner->getSumVarRanges().size(), 3);
        // ASSERT_EQ(inner->getLoopVarRanges().size(), 4);
        // const auto expectedOrder = vector{n, j101, j55, j79};
        // for (int i = 0; i < 4; ++i) {
        //     EXPECT_EQ(inner->getLoopVar(i)->getName(),
        //               expectedOrder[i]->getName());
        // }
    }
}
