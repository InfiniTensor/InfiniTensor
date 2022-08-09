#include "nnet/Pass/Rule5RangeRelaxation.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/visitor.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

// TODO: write a test
// [..net/src/derivator.cc:32 (dfs)] StartDfs = "DFS dep=6" (std::string)
// [..net/src/derivator.cc:32 (dfs)] origin = ==> ROOT
// L<n:0:8><h:0:224><w:0:224><f:0:32>Sum<i13:-1:2><i3:-1:2>  ...
// [i13,i3,n,h,w,f]
//     {L<i13:-1:2><i3:-1:2><n:0:8><h:0:224><w:0:224><f:0:32>Sum  ...  [((1 * w)
//     + (3 * i13)),i13,i13,i3,n,h,w,f]
//     {L<i17:-3:227><i18:-1:2><i13:-1:2><i3:-1:2><n:0:8><h:0:224><w:0:224><f:0:32>Sum
//     ...  [((1 * h) + (3 * i3)),i3,i17,i18,i13,i3,n,h,w,f]
//     {L<i21:-3:227><i22:-1:2><i17:-3:227><i18:-1:2><i13:-1:2><i3:-1:2><n:0:8><h:0:224><w:0:224><f:0:32>Sum<i14:-1:2><i4:-1:2><c:0:16>
//     {({A<pad=0,4,4,0>}[n, c, (i21 + i4), (i14 + i17)] * {K}[f, c, ((3 * i22)
//     + i4), (i14 + (3 * i18))])}}}}
// ==> A : Input Tensor shape=[8,16,224,224] pad=[0,4,4,0]
// ==> K : Input Tensor shape=[32,16,9,9] pad=[0,0,0,0]
//  (nnet::Formula&)
// [..net/src/derivator.cc:670 (rule5RangeRelaxation)] msg = "====== END
// rule5RangeRelaxation: relax iterating ranges i21 (-3,227) to (-1,226), "
// (std::string)
// [..net/src/derivator.cc:32 (dfs)] StartDfs = "DFS dep=7" (std::string)
// [..net/src/derivator.cc:32 (dfs)] origin = ==> ROOT
// L<n:0:8><h:0:224><w:0:224><f:0:32>Sum<i13:-1:2><i3:-1:2>  ...
// [i13,i3,n,h,w,f]
//     {L<i13:-1:2><i3:-1:2><n:0:8><h:0:224><w:0:224><f:0:32>Sum  ...  [((1 * w)
//     + (3 * i13)),i13,i13,i3,n,h,w,f]
//     {L<i17:-3:227><i18:-1:2><i13:-1:2><i3:-1:2><n:0:8><h:0:224><w:0:224><f:0:32>Sum
//     ...  [((1 * h) + (3 * i3)),i3,i17,i18,i13,i3,n,h,w,f]
//     {L<i21:-1:226><i22:-1:2><i17:-3:227><i18:-1:2><i13:-1:2><i3:-1:2><n:0:8><h:0:224><w:0:224><f:0:32><pad=2,0,0,0,0,0,0,0,0,0,>Sum<i14:-1:2><i4:-1:2><c:0:16>
//     {({A<pad=0,4,4,0>}[n, c, (i21 + i4), (i14 + i17)] * {K}[f, c, ((3 * i22)
//     + i4), (i14 + (3 * i18))])}}}}
// ==> A : Input Tensor shape=[8,16,224,224] pad=[0,4,4,0]
// ==> K : Input Tensor shape=[32,16,9,9] pad=[0,0,0,0]
//  (nnet::Formula&)
TEST(Relaxation, NaiveMatch) {
    //     [..rc/nnet/derivator.cc:73 (ruleBasedDerivate)] origin = ==> ROOT
    // L<n:0:8><h:0:224><w:0:224><f:0:32>Sum<i13:0:3><i3:0:3>  ...  [(h + (3 *
    // i3)),i3,(w + (3 * i13)),i13,n,f]
    //     {L<i22:0:230><i23:0:3><i17:0:230><i18:0:3><n:0:8><f:0:32>Sum<i14:0:3><i4:0:3><c:0:16>
    //     {({A<pad=0,0,4,4>}[n, c, ((i22 + i4) + -4), ((i14 + i17) + -4)] *
    //     {K}[f, c, ((3 * i23) + i4), (i14 + (3 * i18))])}}
    // ==> A : Input Tensor shape=[8,16,224,224] pad=[0,0,4,4]
    // ==> K : Input Tensor shape=[32,16,9,9] pad=[0,0,0,0]
    DEFINE_VAR(n);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    DEFINE_VAR(f);
    DEFINE_VAR(i13);
    DEFINE_VAR(i3);
    DEFINE_VAR(i22);
    DEFINE_VAR(i23);
    DEFINE_VAR(i17);
    DEFINE_VAR(i18);
    DEFINE_VAR(i14);
    DEFINE_VAR(i4);
    DEFINE_VAR(c);
    auto A = makeTensor("A", {8, 16, 224, 224}, {0, 0, 4, 4});
    auto K = makeTensor("K", {32, 16, 9, 9});
    auto subA = makeSubscript(A, {n, c, ((i22 + i4) + -4), ((i14 + i17) + -4)});
    auto subK = makeSubscript(K, {f, c, ((3 * i23) + i4), (i14 + (3 * i18))});
    auto innerRange = makeRangeOperator(
        {{i22, {0, 230}},
         {i23, {0, 3}},
         {i17, {0, 230}},
         {i18, {0, 3}},
         {n, {0, 8}},
         {f, {0, 32}}},
        {{i14, {0, 3}}, {i4, {0, 3}}, {c, {0, 16}}}, subA * subK);
    auto subOuter = makeSubscript(
        innerRange, {(h + (3 * i3)), i3, (w + (3 * i13)), i13, n, f});
    auto outerRange = makeRangeOperator(
        {{n, {0, 8}}, {h, {0, 224}}, {w, {0, 224}}, {f, {0, 32}}},
        {{i13, {0, 3}}, {i3, {0, 3}}}, subOuter);
    Derivator derivator(0);
    Formula formula(innerRange, 0);
    Rule5RangeRelaxation pass(derivator);
    pass.setEnableLogging(false);
    pass.setEnableDebug(true);
    auto ret = pass.rule5RangeRelaxation(formula, 0, formula.root);
    ASSERT_TRUE(ret);
    auto rangeOp = as<RangeOpNode>(ret);
    EXPECT_EQ(rangeOp->getRange(i22), pair(2, 228));
    EXPECT_EQ(rangeOp->getRange(i17), pair(2, 228));
}