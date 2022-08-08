#include "nnet/Visitor/AsTVMVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(Conv2conv, 9x9_RuleBased) {
    DEFINE_VAR(i19);
    DEFINE_VAR(i20);
    DEFINE_VAR(i15);
    DEFINE_VAR(i16);
    DEFINE_VAR(n);
    DEFINE_VAR(f);
    auto T2 = make_ref<TensorNode>("T2", vector<int>({8, 288, 226, 226}));
    auto S1 = makeRangeOperator(
        {{i19, {-1, 225}},
         {i20, {-1, 2}},
         {i15, {-1, 225}},
         {i16, {-1, 2}},
         {n, {0, 8}},
         {f, {0, 32}}},
        {}, makeSubscript(T2, {n, 9 * f + 3 * i16 + i20, i15 + 1, i19 + 1}));
    S1->setPaddings({2, 0, 2, 0, 0, 0});
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    DEFINE_VAR(i13);
    DEFINE_VAR(i3);
    auto S2 = makeRangeOperator(
        {{n, {0, 8}}, {h, {0, 224}}, {w, {0, 224}}, {f, {0, 32}}},
        {{i13, {-1, 2}}, {i3, {-1, 2}}},
        makeSubscript(S1, {w + 3 * i13, i13, h + 3 * i3, i3, n, f}));
    std::cout << S2->toReadable() << std::endl;

    AsTVMVisitor visitor;
    visitor.dispatch(S2);
    std::cout << visitor.getStmts() << std::endl;
}
