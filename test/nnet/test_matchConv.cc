#include "nnet/Visitor/PatternMatcher.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "nnet/permutation.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

VecExpr matchConv(Derivator &derivator, const RangeOp &rangeOp) {
    const auto &patternIT = ConvPattern::getPattern();
    return PatternMatcher(derivator, rangeOp)
        .matchWithPattern(rangeOp, patternIT);
}

TEST(MatchConv, Permutation_Generator) {
    DEFINE_VAR(a);
    DEFINE_VAR(b);
    DEFINE_VAR(c);
    DEFINE_VAR(d);
    DEFINE_VAR(e);
    DEFINE_VAR(f);
    DEFINE_VAR(i0);
    DEFINE_VAR(i1);
    DEFINE_VAR(i2);
    PermutationGenerator gen({{a, b, c}, {d, e, f}},
                             {{i0, i1, i2}, {i0, i1, i2}});
    int cnt = 0;
    do {
        if (cnt == 6) {
            auto mapping = gen.get();
            EXPECT_EQ(mapping[a]->getName(), "i0");
            EXPECT_EQ(mapping[b]->getName(), "i2");
            EXPECT_EQ(mapping[c]->getName(), "i1");
            EXPECT_EQ(mapping[d]->getName(), "i0");
            EXPECT_EQ(mapping[e]->getName(), "i1");
            EXPECT_EQ(mapping[f]->getName(), "i2");
        }
        if (cnt == 7) {
            auto mapping = gen.get();
            EXPECT_EQ(mapping[a]->getName(), "i0");
            EXPECT_EQ(mapping[b]->getName(), "i2");
            EXPECT_EQ(mapping[c]->getName(), "i1");
            EXPECT_EQ(mapping[d]->getName(), "i0");
            EXPECT_EQ(mapping[e]->getName(), "i2");
            EXPECT_EQ(mapping[f]->getName(), "i1");
        }
        ++cnt;
    } while (gen.next());
    EXPECT_EQ(cnt, 6 * 6);
}

TEST(MatchConv, NoBatch) {
    DEFINE_VAR(n);
    DEFINE_VAR(c);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    DEFINE_VAR(f);
    DEFINE_VAR(r);
    DEFINE_VAR(s);
    int N = 8, C = 12, H = 224, W = 224, F = 16, R = 3, S = 3;
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, 0, R / 2, S / 2});
    auto B = make_ref<TensorNode>("B", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {n, c, h + r, w + s});
    auto subB = makeSubscript(B, {f, c, r, s});
    auto rangeOp = makeRangeOperator(
        {{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
        {{c, {0, C}}, {r, {-R / 2, R / 2}}, {s, {-S / 2, S / 2}}}, subA * subB);

    // Derivation
    Formula matmul(rangeOp, 0);
    Derivator derivator;
    auto results = matchConv(derivator, rangeOp);
    dbg(results);
    EXPECT_EQ(results.size(), 1u);
    auto tensor = as<TensorNode>(results[0]);
    ASSERT_NE(tensor, nullptr);
    dbg(tensor->getSource()->toReadable());
    dbg(tensor->getSource());

    const auto &conv = as<ConvNode>(tensor->getSource());
    ASSERT_NE(conv, nullptr);
    // Conv{p = 1, 1, s= 1, 1, d= 1, 1; A K = A<pad=0,0,1,1>, B}
    ConvNode matchedConv = ConvNode(rangeOp, A, B, 1, 1);
    EXPECT_EQ(matchedConv, *conv);
}

// wrong index of kernel
TEST(MatchConv, Wrong0) {
    DEFINE_VAR(n);
    DEFINE_VAR(c);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    DEFINE_VAR(f);
    DEFINE_VAR(r);
    DEFINE_VAR(s);
    int N = 8, C = 12, H = 224, W = 224, F = 16, R = 3, S = 3;
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, 0, R / 2, S / 2});
    auto B = make_ref<TensorNode>("B", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {n, c, h + r, w + s});
    auto subB = makeSubscript(B, {c, f, r, s});
    auto rangeOp =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subB);

    // Derivation
    Formula matmul(rangeOp, 0);
    Derivator derivator;
    auto results = matchConv(derivator, rangeOp);
    dbg(results);
    EXPECT_EQ(results.size(), 0u);
}

// wrong index of input tensor
TEST(MatchConv, Wrong1) {
    DEFINE_VAR(n);
    DEFINE_VAR(c);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    DEFINE_VAR(f);
    DEFINE_VAR(r);
    DEFINE_VAR(s);
    int N = 8, C = 12, H = 224, W = 224, F = 16, R = 3, S = 3;
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, 0, R / 2, S / 2});
    auto B = make_ref<TensorNode>("B", vector<int>({F, C, R, S}));

    auto subA = makeSubscript(A, {n, c, h + 2 * r, w + s});
    auto subB = makeSubscript(B, {f, c, r, s});
    auto rangeOp =
        makeRangeOperator({{n, {0, N}}, {f, {0, F}}, {h, {0, H}}, {w, {0, W}}},
                          {{c, {0, C}}, {r, {0, R}}, {s, {0, S}}}, subA * subB);

    // Derivation
    Formula matmul(rangeOp, 0);
    Derivator derivator;
    auto results = matchConv(derivator, rangeOp);
    dbg(results);
    EXPECT_EQ(results.size(), 0u);
}
