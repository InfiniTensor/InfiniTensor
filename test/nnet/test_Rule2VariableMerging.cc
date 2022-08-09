#include "nnet/Pass/Rule2VariableMerging.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

// L<x:0:2><y:0:4>Sum  ...  [(x + (2 * y))]
//     {L<i3:0:8>Sum<t1:0:5>
//     {({A}[i3] * {B}[t1])}}
Expr buildAnsPosPos() {
    DEFINE_VAR(x);
    DEFINE_VAR(y);
    DEFINE_VAR(t1);
    DEFINE_VAR(i3);
    auto A = makeTensor("A", {8});
    auto B = makeTensor("B", {8});
    auto subA = makeSubscript(A, {i3});
    auto subB = makeSubscript(B, {t1});
    auto innerRange =
        makeRangeOperator({{i3, {0, 8}}}, {{t1, {0, 5}}}, subA * subB);
    auto subInner = makeSubscript(innerRange, {x + 2 * y});
    auto outerRange =
        makeRangeOperator({{x, {0, 2}}, {y, {0, 4}}}, {}, subInner);
    return outerRange;
}

void realTest(const Expr &range, const Expr &ans) {
    Derivator derivator(0);
    Rule2VariableMerging pass(derivator);
    Formula origin(range, 0);
    pass.setEnableLogging(false);
    pass.setEnableDebug(true);

    pass.run(origin, 0, origin.root);
    dbg(origin);
    // auto ans = buildAnsPosPos();
    auto hashAns = HashVisitor().dispatch(ans);
    int cntEqual = 0;
    for (const auto &expr : pass.getTransformations()) {
        auto hashExpr = HashVisitor().dispatch(expr);
        if (hashExpr == hashAns)
            ++cntEqual;
        dbg(expr);
    }
    EXPECT_EQ(cntEqual, 1);
}

TEST(Rule2, PosPos) {
    const int a = 1, b = 2;
    DEFINE_VAR(x);
    DEFINE_VAR(y);
    DEFINE_VAR(t1);
    auto A = makeTensor("A", {8});
    auto B = makeTensor("B", {8});
    auto subA = makeSubscript(A, {a * x + b * y});
    auto subB = makeSubscript(B, {t1});
    auto range = makeRangeOperator({{x, {0, 2}}, {y, {0, 4}}}, {{t1, {0, 5}}},
                                   subA * subB);
    realTest(range, buildAnsPosPos());
}

// L<x:0:2><y:0:4>Sum  ...  [((x + (-2 * y)) + 6)]
//     {L<i1:0:8>Sum<t1:0:5>
//     {({A}[i1] * {B}[t1])}}}
Expr buildAnsPosNeg() {
    DEFINE_VAR(x);
    DEFINE_VAR(y);
    DEFINE_VAR(t1);
    DEFINE_VAR(i3);
    auto A = makeTensor("A", {8});
    auto B = makeTensor("B", {8});
    auto subA = makeSubscript(A, {i3});
    auto subB = makeSubscript(B, {t1});
    auto innerRange =
        makeRangeOperator({{i3, {0, 8}}}, {{t1, {0, 5}}}, subA * subB);
    auto subInner = makeSubscript(innerRange, {x - 2 * y + 6});
    auto outerRange =
        makeRangeOperator({{x, {0, 2}}, {y, {0, 4}}}, {}, subInner);
    return outerRange;
}

TEST(Rule2, PosNeg) {
    const int a = 1, b = -2;
    DEFINE_VAR(x);
    DEFINE_VAR(y);
    DEFINE_VAR(t1);
    auto A = makeTensor("A", {8});
    auto B = makeTensor("B", {8});
    auto subA = makeSubscript(A, {a * x + b * y + 6});
    auto subB = makeSubscript(B, {t1});
    auto range = makeRangeOperator({{x, {0, 2}}, {y, {0, 4}}}, {{t1, {0, 5}}},
                                   subA * subB);
    realTest(range, buildAnsPosNeg());
}

// L<x:0:2><y:0:4>Sum  ...  [(((-1 * x) + (-2 * y)) + 7)]
//     {L<i1:0:8>Sum<t1:0:5>
//     {({A}[i1] * {B}[t1])}}
Expr buildAnsNegNeg() {
    DEFINE_VAR(x);
    DEFINE_VAR(y);
    DEFINE_VAR(t1);
    DEFINE_VAR(i3);
    auto A = makeTensor("A", {8});
    auto B = makeTensor("B", {8});
    auto subA = makeSubscript(A, {i3});
    auto subB = makeSubscript(B, {t1});
    auto innerRange =
        makeRangeOperator({{i3, {0, 8}}}, {{t1, {0, 5}}}, subA * subB);
    auto subInner = makeSubscript(innerRange, {(-1) * x - 2 * y + 7});
    auto outerRange =
        makeRangeOperator({{x, {0, 2}}, {y, {0, 4}}}, {}, subInner);
    return outerRange;
}

TEST(Rule2, NegNeg) {
    const int a = -1, b = -2;
    DEFINE_VAR(x);
    DEFINE_VAR(y);
    DEFINE_VAR(t1);
    auto A = makeTensor("A", {8});
    auto B = makeTensor("B", {8});
    auto subA = makeSubscript(A, {a * x + b * y + 7});
    auto subB = makeSubscript(B, {t1});
    auto range = makeRangeOperator({{x, {0, 2}}, {y, {0, 4}}}, {{t1, {0, 5}}},
                                   subA * subB);
    realTest(range, buildAnsNegNeg());
}