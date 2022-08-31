#include "nnet/Visitor/HashVisitor.h"
#include "nnet/derivator.h"
#include "nnet/expr.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;

TEST(Hash, Conv2gemm) {
    int N = 8, H = 224, W = 224, C = 16, F = 32;
    int R = 3, S = 3;
    auto n = make_ref<VarNode>("n");
    auto c = make_ref<VarNode>("c");
    auto h = make_ref<VarNode>("h");
    auto w = make_ref<VarNode>("w");
    auto f = make_ref<VarNode>("f");
    auto r = make_ref<VarNode>("r");
    auto s = make_ref<VarNode>("s");
    auto A = make_ref<TensorNode>("A", vector<int>({N, H, W, C}),
                                  vector<int>{0, R / 2, S / 2, 0});
    auto K = make_ref<TensorNode>("K", vector<int>({R, S, F, C}));

    auto subA = makeSubscript(A, {n, h + r, w + s, c});
    auto subK = makeSubscript(K, {r, s, f, c});
    auto range = makeRangeOperator(
        {{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
        {{c, {0, C}}, {r, {-R / 2, R / 2 + 1}}, {s, {-S / 2, S / 2 + 1}}},
        subA * subK);
    cout << range->toReadable() << endl;
    auto hash0 = HashVisitor().getHash(range);
    cout << hash0 << endl;

    subA = makeSubscript(A, {n, h + s, w + r, c});
    subK = makeSubscript(K, {s, r, f, c});
    range = makeRangeOperator(
        {{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
        {{c, {0, C}}, {r, {-R / 2, R / 2 + 1}}, {s, {-S / 2, S / 2 + 1}}},
        subA * subK);
    cout << range->toReadable() << endl;
    auto hash1 = HashVisitor().getHash(range);
    cout << hash1 << endl;

    subA = makeSubscript(A, {n, s + h, w + r, c});
    subK = makeSubscript(K, {s, r, f, c});
    range = makeRangeOperator(
        {{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
        {{c, {0, C}}, {r, {-R / 2, R / 2 + 1}}, {s, {-S / 2, S / 2 + 1}}},
        subA * subK);
    cout << range->toReadable() << endl;
    auto hash2 = HashVisitor().getHash(range);
    cout << hash2 << endl;

    subA = makeSubscript(A, {n, s + h, w, c});
    subK = makeSubscript(K, {s, r, f, c});
    range = makeRangeOperator(
        {{n, {0, N}}, {h, {0, H}}, {w, {0, W}}, {f, {0, F}}},
        {{c, {0, C}}, {r, {-R / 2, R / 2 + 1}}, {s, {-S / 2, S / 2 + 1}}},
        subA * subK);
    cout << range->toReadable() << endl;
    auto hash3 = HashVisitor().getHash(range);
    cout << hash3 << endl;

    EXPECT_EQ(hash0, hash1);
    EXPECT_EQ(hash0, hash2);
    EXPECT_NE(hash0, hash3);
}
