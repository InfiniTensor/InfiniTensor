#include "nnet/Visitor/CountRoutineVisitor.h"
#include "nnet/Visitor/SimplifyExprVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(Simplify, NaiveMatch) {
    DEFINE_VAR(i14);
    DEFINE_VAR(i15);

    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr = (i15 - i15) + i14;
        auto s = simplifyExprMutator.simplify(expr);
        Var var = as<VarNode>(s);
        ASSERT_TRUE(var);
        EXPECT_TRUE(var->getName() == i14);
    }
    {
        auto expr = (i15 - i15) + i14;
        auto s = simplifyExprMutator.simplify(expr);
        Var var = as<VarNode>(s);
        ASSERT_TRUE(var);
        EXPECT_TRUE(var->getName() == i14);
    }
}

TEST(Simplify, caseInConv2Conv) {
    DEFINE_VAR(i4);
    DEFINE_VAR(i15);
    DEFINE_VAR(i16);
    // cout << range->toReadable() << endl;

    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr1 = ((2 * i15 - (3 * i16)) + ((3 * i16) + i4));
        auto s = simplifyExprMutator.simplify(expr1);
        auto root = as<BinaryOpNode>(s);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->getOpType(), OpType::Add);
        auto t1 = as<BinaryOpNode>(root->getLhs());
        auto t2 = as<BinaryOpNode>(root->getRhs());
        ASSERT_TRUE((t1 == nullptr) ^ (t2 == nullptr));
        BinaryOp bop;
        Var var;
        if (t1) {
            bop = t1;
            var = as<VarNode>(root->getRhs());
        } else {
            bop = t2;
            var = as<VarNode>(root->getLhs());
        }
        EXPECT_EQ(var->getName(), i4->getName());
        Constant constant = as<ConstantNode>(bop->getLhs());
        Var var2 = as<VarNode>(bop->getRhs());
        EXPECT_EQ(constant->getValue(), 2);
        EXPECT_EQ(var2->getName(), i15->getName());
    }
}

TEST(Simplify, caseInSG2BMM) {
    DEFINE_VAR(i6);
    // cout << range->toReadable() << endl;

    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr1 = ((2500 * (i6 / 2500) + ((i6 % 2500))));
        auto s = simplifyExprMutator.simplify(expr1);
        dbg(s);
        auto root = as<VarNode>(s);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->getName(), i6->getName());
    }
}

TEST(Simplify, AdvancedDivMod) {
    DEFINE_VAR(i7);
    // cout << range->toReadable() << endl;

    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr1 = ((5000 * (i7 / 2500) + 2 * ((i7 % 2500))));
        auto s = simplifyExprMutator.simplify(expr1);
        dbg(s);
        auto root = as<BinaryOpNode>(s);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->getOpType(), OpType::Mul);
        auto t1 = as<ConstantNode>(root->getLhs());
        auto t2 = as<ConstantNode>(root->getRhs());
        ASSERT_TRUE((t1 == nullptr) ^ (t2 == nullptr));
        Constant bop;
        Var var;
        if (t1) {
            bop = t1;
            var = as<VarNode>(root->getRhs());
        } else {
            bop = t2;
            var = as<VarNode>(root->getLhs());
        }
        EXPECT_EQ(var->getName(), i7->getName());
        EXPECT_EQ(bop->getValue(), 2);
    }
}

TEST(Simplify, AdvancedDivMod2) {
    DEFINE_VAR(i4);
    DEFINE_VAR(i15);
    DEFINE_VAR(i16);
    // cout << range->toReadable() << endl;

    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr1 =
            ((2 * i15 - (3 * i16)) + (9 * (i16 / 3) + 3 * (i16 % 3) + i4));
        auto s = simplifyExprMutator.simplify(expr1);
        auto root = as<BinaryOpNode>(s);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->getOpType(), OpType::Add);
        auto t1 = as<BinaryOpNode>(root->getLhs());
        auto t2 = as<BinaryOpNode>(root->getRhs());
        ASSERT_TRUE((t1 == nullptr) ^ (t2 == nullptr));
        BinaryOp bop;
        Var var;
        if (t1) {
            bop = t1;
            var = as<VarNode>(root->getRhs());
        } else {
            bop = t2;
            var = as<VarNode>(root->getLhs());
        }
        EXPECT_EQ(var->getName(), i4->getName());
        Constant constant = as<ConstantNode>(bop->getLhs());
        Var var2 = as<VarNode>(bop->getRhs());
        EXPECT_EQ(constant->getValue(), 2);
        EXPECT_EQ(var2->getName(), i15->getName());
    }
}

TEST(Simplify, Constant) {
    DEFINE_VAR(i14);
    DEFINE_VAR(i15);

    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr = (i15 - i15) + i14 + 1;
        auto s = simplifyExprMutator.simplify(expr);
        dbg(expr, s);
        auto binaryOp = as<BinaryOpNode>(s);
        ASSERT_TRUE(binaryOp);
        auto ca = as<ConstantNode>(binaryOp->getLhs());
        auto cb = as<ConstantNode>(binaryOp->getRhs());
        EXPECT_TRUE(!ca ^ !cb);
        if (ca != nullptr)
            EXPECT_EQ(ca->getValue(), 1);
        else
            EXPECT_EQ(cb->getValue(), 1);
        // EXPECT_TRUE(var->getName() == i14);
    }
    {
        auto expr = -3 + (i15 - i15) + i14 + 1;
        auto s = simplifyExprMutator.simplify(expr);
        dbg(expr, s);
        auto binaryOp = as<BinaryOpNode>(s);
        ASSERT_TRUE(binaryOp);
        auto ca = as<ConstantNode>(binaryOp->getLhs());
        auto cb = as<ConstantNode>(binaryOp->getRhs());
        EXPECT_TRUE(!ca ^ !cb);
        int finalConst = -2;
        if (ca != nullptr)
            EXPECT_EQ(ca->getValue(), finalConst);
        else
            EXPECT_EQ(cb->getValue(), finalConst);
    }
}

TEST(Simplify, AdvancedDivMod3Negative_TConv) {
    DEFINE_VAR(i5);
    // cout << range->toReadable() << endl;

    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr1 = ((1 * (i5 % -2)) + (2 * (i5 / -2)));
        auto s = simplifyExprMutator.simplify(expr1);
        dbg(s);
        auto root = as<VarNode>(s);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->getName(), "i5");
    }
}

TEST(Simplify, SingleDivOrMod_TConv) {
    DEFINE_VAR(i5);
    SimplifyExprVisitor simplifyExprMutator;
    {
        auto expr1 = 1 * (i5 / 2);
        auto s = simplifyExprMutator.simplify(expr1);
        dbg(s);
        auto root = as<BinaryOpNode>(s);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->getOpType(), OpType::Div);
        auto var = as<VarNode>(root->getLhs());
        auto divisor = as<ConstantNode>(root->getRhs());
        EXPECT_EQ(var, "i5");
        EXPECT_EQ(divisor->getValue(), 2);
    }
    {
        auto expr1 = 1 * (i5 % 2);
        auto s = simplifyExprMutator.simplify(expr1);
        dbg(s);
        auto root = as<BinaryOpNode>(s);
        ASSERT_TRUE(root);
        EXPECT_EQ(root->getOpType(), OpType::Mod);
        auto var = as<VarNode>(root->getLhs());
        auto divisor = as<ConstantNode>(root->getRhs());
        EXPECT_EQ(var, "i5");
        EXPECT_EQ(divisor->getValue(), 2);
    }
}