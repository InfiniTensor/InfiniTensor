#include "nnet/dlt.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(DLT, Simple) {
    DEFINE_VAR(c);
    DEFINE_VAR(f);
    DEFINE_VAR(p1);
    DEFINE_VAR(p2);
    DEFINE_VAR(q1);
    DEFINE_VAR(q2);
    int C = 12, F = 16, R = 9, S = 9;
    auto A = make_ref<TensorNode>("A", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {f, c, 3 * p1 + p2, 3 * q1 + q2});
    auto rangeOp =
        makeRangeOperator({{p1, {0, 3}}, {q1, {0, 3}}, {f, {0, F}}},
                          {{c, {0, C}}, {p2, {0, 3}}, {q2, {0, 3}}}, subA);
    DLT dlt;
    dlt.split(2, 3);
    auto opt = dlt.apply(rangeOp, subA, "dltedA");
    ASSERT_TRUE(opt.has_value());
    auto sub = as<SubscriptNode>(*opt);
    dbg(rangeOp, sub);
    ASSERT_TRUE(sub != nullptr);
    EXPECT_EQ(sub->getDims(), 5u);
    EXPECT_EQ(sub->getIndex(2)->hash(), p1->hash());
    EXPECT_EQ(sub->getIndex(3)->hash(), p2->hash());
}

TEST(DLT, Conv2Conv) {
    DEFINE_VAR(c);
    DEFINE_VAR(f);
    DEFINE_VAR(p1);
    DEFINE_VAR(p2);
    DEFINE_VAR(q1);
    DEFINE_VAR(q2);
    int C = 12, F = 16, R = 9, S = 9;
    auto A = make_ref<TensorNode>("A", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {f, c, 3 * p1 + p2, 3 * q1 + q2});
    auto rangeOp =
        makeRangeOperator({{p1, {0, 3}}, {q1, {0, 3}}, {f, {0, F}}},
                          {{c, {0, C}}, {p2, {0, 3}}, {q2, {0, 3}}}, subA);
    DLT dlt;
    dlt.split(3, 3);
    dlt.split(2, 3);
    dlt.merge(0, 2);
    dlt.merge(0, 3);
    auto opt = dlt.apply(rangeOp, subA, "dltedA");
    ASSERT_TRUE(opt.has_value());
    auto sub = as<SubscriptNode>(*opt);
    ASSERT_TRUE(sub != nullptr);
    EXPECT_EQ(sub->getDims(), 4u);
}

TEST(DLT, Wrong0) {
    DEFINE_VAR(c);
    DEFINE_VAR(f);
    DEFINE_VAR(p1);
    DEFINE_VAR(p2);
    DEFINE_VAR(q1);
    DEFINE_VAR(q2);
    int C = 12, F = 16, R = 9, S = 9;
    auto A = make_ref<TensorNode>("A", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {f, c, 3 * p1 + p2, 3 * q1 + q2});
    auto rangeOp =
        makeRangeOperator({{p1, {0, 3}}, {q1, {0, 3}}, {f, {0, F}}},
                          {{c, {0, C}}, {p2, {0, 4}}, {q2, {0, 3}}}, subA);
    DLT dlt;
    dlt.split(2, 3);
    auto opt = dlt.apply(rangeOp, subA, "dltedA");
    ASSERT_FALSE(opt.has_value());
}
