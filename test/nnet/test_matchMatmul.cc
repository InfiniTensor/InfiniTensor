#include "nnet/Visitor/FullPrinterVisitor.h"
#include "nnet/Visitor/PatternMatcher.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/iterator_table.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

VecExpr matchMatmul(Derivator &derivator, const RangeOp &rangeOp) {
    const auto &patternIT = MatmulPattern::getMatmulPattern();
    return PatternMatcher(derivator, rangeOp)
        .matchWithPattern(rangeOp, patternIT);
}

TEST(MatchMatmul, NoBatch) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("m");
    auto n = make_ref<VarNode>("n");
    auto k = make_ref<VarNode>("k");
    // Transpose requires the existance of source for inputs
    auto _A = make_ref<TensorNode>("A_shadow", vector<int>({M, K}));
    auto _B = make_ref<TensorNode>("B_shadow", vector<int>({N, K}));
    auto rangeA = makeRangeOperator({{m, {0, M}}, {k, {0, K}}}, {},
                                    makeSubscript(_A, {m, k}));
    auto rangeB = makeRangeOperator({{n, {0, N}}, {k, {0, K}}}, {},
                                    makeSubscript(_A, {n, k}));
    auto elemA =
        make_ref<ElementWiseNode>(rangeA, vector<Tensor>{_A}, _A->getShape());
    auto elemB =
        make_ref<ElementWiseNode>(rangeB, vector<Tensor>{_B}, _B->getShape());
    auto A = makeTensor("A", vector<int>({M, K}), {}, elemA);
    auto B = makeTensor("B", vector<int>({N, K}), {}, elemB);

    auto subA = makeSubscript(A, {m, k});
    auto subB = makeSubscript(B, {n, k});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
                                   subA * subB);

    // Derivation
    Formula matmul(range, 0);
    Derivator derivator;
    auto results = matchMatmul(derivator, range);
    // Matmul{bmnk = 1, 224, 8, 16; AB = A, B; transAB = 0, 0}
    // Matmul{bmnk = 1, 8, 224, 16; AB = B, A; transAB = 0, 0}
    vector<MatmulNode> answers = {
        MatmulNode(range, A, B, 1, 224, 8, 16, false, true)};
    set<MatmulArgs> argSet;
    for (const auto &result : results) {
        static int cnt = 0;
        cout << "========" << ++cnt << endl;
        std::cout << FullPrinterVisitor().print(result);

        Tensor tensor = as<TensorNode>(result);
        if (!tensor) {
            tensor = as<TensorNode>(
                as<SubscriptNode>(as<RangeOpNode>(result)->getSummand())
                    ->getObject());
        }
        argSet.emplace(as<MatmulNode>(tensor->getSource())->getArgs());
    }
    EXPECT_EQ(results.size(), 8u);
    EXPECT_EQ(argSet.size(), 8u);
    EXPECT_TRUE(argSet.count({1, 224, 8, 16, false, true}));
    EXPECT_TRUE(argSet.count({1, 8, 224, 16, false, true}));
}

TEST(MatchMatmul, Illegal0) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("m");
    auto n = make_ref<VarNode>("n");
    auto k = make_ref<VarNode>("k");
    auto A = make_ref<TensorNode>("A", vector<int>({M, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {m, k});
    auto subB = makeSubscript(B, {k, k});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
                                   subA * subB);
    // Derivation
    Formula matmul(range, 0);
    Derivator derivator;
    auto results = matchMatmul(derivator, range);
    vector<MatmulNode> answers = {};
    EXPECT_EQ(results.size(), answers.size());
}

TEST(MatchMatmul, Illegal1) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("m");
    auto n = make_ref<VarNode>("n");
    auto k = make_ref<VarNode>("k");
    auto A = make_ref<TensorNode>("A", vector<int>({M, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {m, k});
    auto subB = makeSubscript(B, {n, k});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}, {k, {0, K}}}, {},
                                   subA * subB);
    // Derivation
    Formula matmul(range, 0);
    Derivator derivator;
    auto results = matchMatmul(derivator, range);
    vector<MatmulNode> answers = {};
    EXPECT_EQ(results.size(), answers.size());
}

TEST(MatchMatmul, Illegal2) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("m");
    auto n = make_ref<VarNode>("n");
    auto k = make_ref<VarNode>("k");
    auto A = make_ref<TensorNode>("A", vector<int>({M, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {m, m + k});
    auto subB = makeSubscript(B, {n, k});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
                                   subA * subB);
    // Derivation
    Formula matmul(range, 0);
    Derivator derivator;
    auto results = matchMatmul(derivator, range);
    // dbg(results.size());
    // for (const auto &result : results) {
    //     dbg(result);
    //     dbg(*result);
    //     dbg(as<TensorNode>(result)->getShape());
    //     dbg(as<TensorNode>(result)->getSource());
    // }
    vector<MatmulNode> answers = {};
    EXPECT_EQ(results.size(), answers.size());
}

TEST(MatchMatmul, Illegal3) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("m");
    auto n = make_ref<VarNode>("n");
    auto k = make_ref<VarNode>("k");
    auto A = make_ref<TensorNode>("A", vector<int>({M, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {m, n + k});
    auto subB = makeSubscript(B, {n, k});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
                                   subA * subB);
    // Derivation
    Formula matmul(range, 0);
    Derivator derivator;
    auto results = matchMatmul(derivator, range);
    // dbg(results.size());
    // for (const auto &result : results) {
    //     dbg(result);
    //     dbg(*result);
    //     dbg(as<TensorNode>(result)->getShape());
    //     dbg(as<TensorNode>(result)->getSource());
    // }
    vector<MatmulNode> answers = {};
    EXPECT_EQ(results.size(), answers.size());
}

// Different position of the appearance
TEST(MatchMatmul, Illegal4) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("m");
    auto n = make_ref<VarNode>("n");
    auto k = make_ref<VarNode>("k");
    auto A = make_ref<TensorNode>("A", vector<int>({M, K}));
    auto B = make_ref<TensorNode>("B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {m, k});
    auto subB = makeSubscript(B, {k, n});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
                                   subA * subB);
    // Derivation
    Formula matmul(range, 0);
    Derivator derivator;
    auto results = matchMatmul(derivator, range);
    // dbg(results.size());
    // for (const auto &result : results) {
    //     dbg(result);
    //     dbg(*result);
    //     dbg(as<TensorNode>(result)->getShape());
    //     dbg(as<TensorNode>(result)->getSource());
    // }
    vector<MatmulNode> answers = {};
    EXPECT_EQ(results.size(), answers.size());
}

// Different position of the appearance
TEST(MatchMatmul, IteratorTable1) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("_m");
    auto n = make_ref<VarNode>("_n");
    auto k = make_ref<VarNode>("_k");
    auto A = make_ref<TensorNode>("_A", vector<int>({M, K}));
    auto B = make_ref<TensorNode>("_B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {m, k});
    auto subB = makeSubscript(B, {n, k});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
                                   subA * subB);
    class IteratorTable exprIT;
    ASSERT_TRUE(exprIT.analyzeExpr(range));
    exprIT.buildTable({0, 1});
    auto const &[posTable, iterInTensorDim, strideInTensor] =
        exprIT.getTables();
    // dbg(posTable, iterInTensorDim, strideInTensor);
    EXPECT_EQ(posTable.size(), 8u);
    for (int i = 0; i < 8; ++i) {
        if (i == 3 || i == 5 || i == 6)
            EXPECT_EQ(posTable[i].size(), 1u);
        else
            EXPECT_EQ(posTable[i].size(), 0u);
    }
    // iterInTensorDim = {{{"_m"}, {"_k"}}, {{"_n"}, {"_k"}}}
    EXPECT_EQ(iterInTensorDim.size(), 2u);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(iterInTensorDim[i].size(), 2u);
        for (int j = 0; j < 2; ++j)
            EXPECT_EQ(iterInTensorDim[i][j].size(), 1u);
    }
    EXPECT_TRUE(iterInTensorDim[0][0][0]->equal(m));
    EXPECT_TRUE(iterInTensorDim[0][1][0]->equal(k));
    EXPECT_TRUE(iterInTensorDim[1][0][0]->equal(n));
    EXPECT_TRUE(iterInTensorDim[0][1][0]->equal(k));
    // strideInTensor = {{"_k", {1, 1}}, {"_m", {16, 0}}, {"_n", {0, 16}}}
    EXPECT_EQ(strideInTensor.size(), 3u);
    EXPECT_EQ(strideInTensor.at(k)[0], 1);
    EXPECT_EQ(strideInTensor.at(k)[1], 1);
    EXPECT_EQ(strideInTensor.at(m)[0], 16);
    EXPECT_EQ(strideInTensor.at(m)[1], 0);
    EXPECT_EQ(strideInTensor.at(n)[0], 0);
    EXPECT_EQ(strideInTensor.at(n)[1], 16);
}

// Different position of the appearance
TEST(MatchMatmul, IteratorTable2) {
    int M = 224, N = 8, K = 16;
    auto m = make_ref<VarNode>("_m");
    auto n = make_ref<VarNode>("_n");
    auto k = make_ref<VarNode>("_k");
    auto c2 = make_ref<ConstantNode>(2);
    auto A = make_ref<TensorNode>("_A", vector<int>({M, K}));
    auto B = make_ref<TensorNode>("_B", vector<int>({N, K}));

    auto subA = makeSubscript(A, {m, k + m});
    auto subB = makeSubscript(B, {n, c2 * (k + c2)});
    auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
                                   subA * subB);
    class IteratorTable exprIT;
    ASSERT_TRUE(exprIT.analyzeExpr(range));
    exprIT.buildTable({0, 1});
    auto const &[posTable, iterInTensorDim, strideInTensor] =
        exprIT.getTables();
    // dbg(posTable, iterInTensorDim, strideInTensor);
    EXPECT_EQ(posTable.size(), 8u);
    for (int i = 0; i < 8; ++i) {
        if (i == 3 || i == 5 || i == 6)
            EXPECT_EQ(posTable[i].size(), 1u);
        else
            EXPECT_EQ(posTable[i].size(), 0u);
    }
    // iterInTensorDim = {{{"_m"}, {"_k"}}, {{"_n"}, {"_k"}}}
    EXPECT_EQ(iterInTensorDim.size(), 2u);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(iterInTensorDim[i].size(), 2u);
        for (int j = 0; j < 2; ++j)
            if (i == 0 && j == 1)
                EXPECT_EQ(iterInTensorDim[i][j].size(), 2u);
            else
                EXPECT_EQ(iterInTensorDim[i][j].size(), 1u);
    }
    EXPECT_TRUE(iterInTensorDim[0][0][0]->equal(m));
    EXPECT_TRUE(iterInTensorDim[0][1][0]->equal(k));
    EXPECT_TRUE(iterInTensorDim[1][0][0]->equal(n));
    EXPECT_TRUE(iterInTensorDim[0][1][0]->equal(k));
    // strideInTensor = {{"_k", {1, 1}}, {"_m", {16, 0}}, {"_n", {0, 16}}}
    EXPECT_EQ(strideInTensor.size(), 3u);
    EXPECT_EQ(strideInTensor.at(k)[0], 1);
    EXPECT_EQ(strideInTensor.at(k)[1], 2);
    EXPECT_EQ(strideInTensor.at(m)[0], 17);
    EXPECT_EQ(strideInTensor.at(m)[1], 0);
    EXPECT_EQ(strideInTensor.at(n)[0], 0);
    EXPECT_EQ(strideInTensor.at(n)[1], 16);
}

// TEST(MatchMatmul, NoBatch_Traspose) {
//     int M = 224, N = 8, K = 16;
//     auto m = make_ref<VarNode>("m");
//     auto n = make_ref<VarNode>("n");
//     auto k = make_ref<VarNode>("k");
//     auto A = make_ref<TensorNode>("A", vector<int>({M, K}));
//     auto B = make_ref<TensorNode>("B", vector<int>({N, K}));

//     auto subA = makeSubscript(A, {m, k});
//     auto subB = makeSubscript(B, {n, k});
//     auto rangeA = makeRangeOperator({{m, {0, M}}, {k, {0, K}}}, {}, subA);
//     auto rangeB = makeRangeOperator({{n, {0, N}}, {k, {0, K}}}, {}, subB);
//     auto ewA = make_ref<ElementWiseNode>(rangeA, vector<Tensor>{A},
//                                             rangeA->getOutputShape());
//     auto ewB = make_ref<ElementWiseNode>(rangeB, vector<Tensor>{B},
//                                             rangeB->getOutputShape());
//     auto tensorA = makeTensor("TA", A->getShape(), {}, ewA);
//     auto tensorB = makeTensor("TB", B->getShape(), {}, ewB);
//     auto subRangeA = makeSubscript(tensorA, {m, k});
//     auto subRangeB = makeSubscript(tensorB, {n, k});
//     auto range = makeRangeOperator({{m, {0, M}}, {n, {0, N}}}, {{k, {0, K}}},
//                                    subRangeA * subRangeB);

//     // Derivation
//     Formula matmul(range, 0);
//     Derivator derivator;
//     auto results = derivator.matchMatmul(range);
//     // Matmul{bmnk = 1, 224, 8, 16; AB = A, B; transAB = 0, 0}
//     // Matmul{bmnk = 1, 8, 224, 16; AB = B, A; transAB = 0, 0}
//     EXPECT_EQ(results.size(), 8);
//     vector<MatmulNode> answers = {
//         MatmulNode(range, {A, B}, 1, 224, 8, 16, false, false)};
//     // tensor permutation is diabled
//     // MatmulNode(range, {B, A}, 1, 8, 224, 16, false, false)};
//     for (const auto &result : results) {
//         dbg(result);
//         dbg(FullPrinterVisitor().print(result));
//     }
//     // for (const auto &ans : answers) {
//     //     bool matched = false;
//     //     for (const auto &result : results) {
//     //         FullPrinterVisitor().print(result);
//     //         auto resultMatmul = //
//     //         as<MatmulNode>(as<TensorNode>(result)->getSource());
//     //         EXPECT_TRUE(resultMatmul != nullptr);
//     //         if (ans == *resultMatmul)
//     //             matched = true;
//     //     }
//     //     EXPECT_TRUE(matched);
//     // }
// }
