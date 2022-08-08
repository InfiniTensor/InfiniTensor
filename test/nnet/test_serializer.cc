#include "nnet/Visitor/FullPrinterVisitor.h"
#include "nnet/Visitor/Serializer.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

//{L<i3:0:2500><i4:0:4><b:0:8><w:0:65>Sum<k:0:512>
//{({A}[b, (i3 + (2500 * i4)), k] * {B<pad=0,128,0>}[b, ((i3 + (2500 * i4)) +
// w), k])}}
// ==> A : Input Tensor shape=[8,10000,512] pad=[0,0,0]
// ==> B : Input Tensor shape=[8,10000,512] pad=[0,128,0]

Expr buildSimpleExpr() {
    DEFINE_VAR(b);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    DEFINE_VAR(i3);
    DEFINE_VAR(i4);
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
    DEFINE_VAR(j1);
    DEFINE_VAR(j2);
    DEFINE_VAR(j3);
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

    DEFINE_VAR(b);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    DEFINE_VAR(i3);
    DEFINE_VAR(i4);
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
    auto isSuccessful = Serializer().serialize(range, "./test_serializer.json");
    EXPECT_TRUE(isSuccessful);
}

TEST(Serializer, CompareTwoExprs) {
    DEFINE_VAR(b);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    DEFINE_VAR(i3);
    DEFINE_VAR(i4);
    auto A = makeTensor("A", {8, 10000, 512}, {0, 0, 0});
    auto B = makeTensor("B", {8, 10000, 512}, {0, 128, 0});
    auto subA = makeSubscript(A, {b, (i3 + (2500 * i4)), k});
    auto subB = makeSubscript(B, {b, ((i3 + (2500 * i4)) + w), k});
    auto range = makeRangeOperator(
        {{i3, {0, 2500}}, {i4, {0, 4}}, {b, {0, 8}}, {w, {0, 65}}},
        {{k, {0, 512}}}, subA * subB);
    Serializer().serialize(range, "./test_serializer.json");
    auto expr = Serializer().deserialize("./test_serializer.json");

    EXPECT_EQ(range->toReadable(), expr->toReadable());
}

TEST(Serializer, Serialization_NestedTensor) {
    FullPrinterVisitor printer;
    auto range = buildNestedExpr();
    auto ans = printer.print(range);
    dbg(ans);
    auto isSuccessful = Serializer().serialize(range, "./test_serializer.json");
    EXPECT_TRUE(isSuccessful);
    auto exprDeserialized = Serializer().deserialize("./test_serializer.json");
    auto output = printer.print(exprDeserialized);
    dbg(output);
    EXPECT_EQ(output, ans);
}