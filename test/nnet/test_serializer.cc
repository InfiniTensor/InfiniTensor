#include "core/graph.h"
#include "nnet/Visitor/FullPrinterVisitor.h"
#include "nnet/Visitor/Serializer.h"
#include "nnet/test.h"
#include "operators/membound.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

//{L<i3:0:2500><i4:0:4><b:0:8><w:0:65>Sum<k:0:512>
//{({A}[b, (i3 + (2500 * i4)), k] * {B<pad=0,128,0>}[b, ((i3 + (2500 * i4)) +
// w), k])}}
// ==> A : Input Tensor shape=[8,10000,512] pad=[0,0,0]
// ==> B : Input Tensor shape=[8,10000,512] pad=[0,128,0]

Expr buildSimpleExpr() {
    DEFINE_VAR(b, w, k, i3, i4);
    auto A = makeTensor("A", {8, 10000, 512}, {0, 0, 0});
    auto B = makeTensor("B", {8, 10000, 512}, {0, 128, 0});
    auto subA = makeSubscript(A, {b, (i3 + (2500 * i4)), k});
    auto subB = makeSubscript(B, {b, ((i3 + (2500 * i4)) + w), k});
    auto range = makeRangeOperator(
        {{i3, {0, 2500}}, {i4, {0, 4}}, {b, {0, 8}}, {w, {0, 65}}},
        {{k, {0, 512}}}, subA * subB);
    return range;
}

Expr buildNestedExpr() {
    DEFINE_VAR(j1, j2, j3);
    // Build a Matmul to verify.
    const int M = 10000, N = 512, K = 3;
    auto C = make_ref<TensorNode>("C", vector<int>({M, K}));
    auto D = make_ref<TensorNode>("D", vector<int>({N, K}));
    auto F = make_ref<TensorNode>("F", vector<int>({N, K}));
    auto matmulExpr = makeSubscript(C, {j1, j3}) * makeSubscript(D, {j2, j3});
    Expr expr = makeRangeOperator({{j1, {0, M}}, {j2, {0, N}}}, {{j3, {0, K}}},
                                  matmulExpr);
    auto matmul = make_ref<MatmulNode>(expr, C, D, 1, M, N, K, false, false);

    vector<int> shapeE{N, K};
    auto ele2 = make_ref<ElementWiseNode>(Expr(), vector{F}, shapeE);
    auto E = make_ref<TensorNode>("E", shapeE, shapeE, ele2);
    auto ele1 = make_ref<ElementWiseNode>(expr, vector{E}, shapeE);

    DEFINE_VAR(b, w, k, i3, i4);
    auto A = makeTensor("A", {8, 10000, 512}, {0, 0, 0}, matmul);
    auto B = makeTensor("B", {8, 10000, 512}, {0, 128, 0}, ele1);
    auto subA = makeSubscript(A, {b, (i3 + (2500 * i4)), k});
    auto subB = makeSubscript(B, {b, ((i3 + (2500 * i4)) + w), k});
    auto range = makeRangeOperator(
        {{i3, {0, 2500}}, {i4, {0, 4}}, {b, {0, 8}}, {w, {0, 65}}},
        {{k, {0, 512}}}, subA * subB);
    return range;
}

TEST(Serializer, Serialization) {
    auto range = buildSimpleExpr();
    auto isSuccessful = Serializer().toFile(range, "./test_serializer.json");
    EXPECT_TRUE(isSuccessful);
}

TEST(Serializer, CompareTwoExprs) {
    DEFINE_VAR(b, w, k, i3, i4);
    auto A = makeTensor("A", {8, 10000, 512}, {0, 0, 0});
    auto B = makeTensor("B", {8, 10000, 512}, {0, 128, 0});
    auto subA = makeSubscript(A, {b, (i3 + (2500 * i4)), k});
    auto funcA = make_ref<FuncNode>(subA, FuncType::Relu);
    auto subB = makeSubscript(B, {b, ((i3 + (2500 * i4)) + w), k});
    auto range = makeRangeOperator(
        {{i3, {0, 2500}}, {i4, {0, 4}}, {b, {0, 8}}, {w, {0, 65}}},
        {{k, {0, 512}}}, funcA * subB);
    Serializer().toFile(range, "./test_serializer.json");
    auto expr = Serializer().fromFile("./test_serializer.json");
    dbg(expr);

    EXPECT_EQ(range->toReadable(), expr->toReadable());
}

TEST(Serializer, Serialization_NestedTensor) {
    FullPrinterVisitor printer;
    auto range = buildNestedExpr();
    auto ans = printer.print(range);
    auto isSuccessful = Serializer().toFile(range, "./test_serializer.json");
    EXPECT_TRUE(isSuccessful);
    auto exprDeserialized = Serializer().fromFile("./test_serializer.json");
    auto output = printer.print(exprDeserialized);
    EXPECT_EQ(output, ans);
}

TEST(Serializer, Serialization_memboundOp) {
    auto expr = buildSimpleExpr();
    auto A = makeTensor("A", {8, 10000, 512}, {0, 0, 0});
    auto B = makeTensor("B", {8, 10000, 512}, {0, 128, 0});
    // using namespace infini;
    auto runtime = infini::NativeCpuRuntimeObj::getInstance();
    auto g = infini::make_ref<infini::GraphObj>(runtime);
    auto AT = g->addTensor({8, 10000, 512});
    auto BT = g->addTensor({8, 10000, 512});
    auto CT = g->addTensor({2500, 4, 8, 65});

    vector<Tensor> nnetInputs{A, B};
    double execTime = 1;
    string hint = "test";
    infini::MemBoundObj memboundOp(nullptr, {AT, BT}, {CT}, nnetInputs, expr,
                                   execTime, hint);
    auto str = memboundOp.toJson();
    auto [exprLoaded, nnetInputsLoaded, execTimeLoaded, hintLoaded] =
        Serializer().membundOpFromString(str);
    EXPECT_EQ(expr->toReadable(), exprLoaded->toReadable());
    EXPECT_EQ(execTime, execTimeLoaded);
    EXPECT_EQ(nnetInputs.size(), nnetInputsLoaded.size());
    for (size_t i = 0; i < nnetInputs.size(); ++i)
        EXPECT_EQ(nnetInputs[i]->toReadable(),
                  nnetInputsLoaded[i]->toReadable());
    EXPECT_EQ(hint, hintLoaded);
}
