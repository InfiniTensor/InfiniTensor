#include "graph.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/nmutator.h"
#include "operator.h"
#include "search_engine.h"
#include "gtest/gtest.h"
using namespace infini;
using namespace nnet;
using namespace std;

TEST(TransposeOp2Expr, Basic) {
    auto g = new tpm::Graph();
    auto AT = g->tensor({2, 4, 8, 16});
    auto op = new TransposeOp(AT, -1, {3, 1, 2, 0});

    auto i = make_ref<VarNode>("i");
    auto j = make_ref<VarNode>("j");
    auto k = make_ref<VarNode>("k");
    auto l = make_ref<VarNode>("l");
    auto AN = make_ref<TensorNode>("A", vector<int>({2, 4, 6, 8}));
    auto subA = makeSubscript(AN, {l, j, k, i});
    auto ans = makeRangeOperator(
        {{i, {0, 16}}, {j, {0, 4}}, {k, {0, 8}}, {l, {0, 2}}}, {}, subA);
    ASSERT_TRUE(HashVisitor().getHash(ans) ==
                HashVisitor().getHash(tpm::transposeOpToExpression(op)));
}