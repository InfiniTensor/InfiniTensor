#include "nnet/Visitor/Interpreter.h"
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
TEST(Interpreter, SingleStage) {
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
    cout << range->toReadable() << endl;

    auto dataA = make_ref<vector<int>>(8 * 10000 * 512);
    auto dataB = make_ref<vector<int>>(8 * 10000 * 512);
    for (int i = 0; i < 8 * 10000 * 512; i++) {
        dataA->operator[](i) = i;
        dataB->operator[](i) = i;
    }
    unordered_map<string, Ref<vector<int>>> inputs{{"A", dataA}, {"B", dataB}};
    vector<vector<int>> positions{{0, 0, 0, 0}, {1, 2, 3, 4}};
    auto values1 = Interpreter(inputs).interpret(range, positions);
    dbg(values1);
}

//{L<i3:0:2500><i4:0:4><b:0:8><w:0:65>Sum  ...  [(i3 + (2500 * i4)),b,w]
//{L<i45:0:10000><b:0:8><w:0:65>Sum<k:0:512>
//{({A}[b, i45, k] * {B<pad=0,128,0>}[b, (i45 + w), k])}}}
// ==> A : Input Tensor shape=[8,10000,512] pad=[0,0,0]
// ==> B : Input Tensor shape=[8,10000,512] pad=[0,128,0]
TEST(Interpreter, DoubleNestedStages) {
    DEFINE_VAR(b);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    DEFINE_VAR(i3);
    DEFINE_VAR(i4);
    DEFINE_VAR(i45);
    auto A = makeTensor("A", {8, 10000, 512}, {0, 0, 0});
    auto B = makeTensor("B", {8, 10000, 512}, {0, 128, 0});
    auto subA = makeSubscript(A, {b, i45, k});
    auto subB = makeSubscript(B, {b, (i45 + w), k});
    auto innerRange =
        makeRangeOperator({{i45, {0, 10000}}, {b, {0, 8}}, {w, {0, 65}}},
                          {{k, {0, 512}}}, subA * subB);
    auto subOuter = makeSubscript(innerRange, {(i3 + (2500 * i4)), b, w});
    auto outerRange = makeRangeOperator(
        {{i3, {0, 2500}}, {i4, {0, 4}}, {b, {0, 8}}, {w, {0, 65}}}, {},
        subOuter);
    cout << outerRange->toReadable() << endl;

    auto dataA = make_ref<vector<int>>(8 * 10000 * 512);
    auto dataB = make_ref<vector<int>>(8 * 10000 * 512);
    for (int i = 0; i < 8 * 10000 * 512; i++) {
        dataA->operator[](i) = i;
        dataB->operator[](i) = i;
    }
    unordered_map<string, Ref<vector<int>>> inputs{{"A", dataA}, {"B", dataB}};
    vector<vector<int>> positions{{0, 0, 0, 0}, {1, 2, 3, 4}};
    auto values2 = Interpreter(inputs).interpret(outerRange, positions);
    dbg(values2);
}

// The above two expressions
TEST(Interpreter, CompareTwoExprs) {
    DEFINE_VAR(b);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    DEFINE_VAR(i3);
    DEFINE_VAR(i4);
    DEFINE_VAR(i45);
    auto A = makeTensor("A", {8, 10000, 512}, {0, 0, 0});
    auto B = makeTensor("B", {8, 10000, 512}, {0, 128, 0});
    // singleStage
    auto subA1 = makeSubscript(A, {b, (i3 + (2500 * i4)), k});
    auto subB1 = makeSubscript(B, {b, ((i3 + (2500 * i4)) + w), k});
    auto range = makeRangeOperator(
        {{i3, {0, 2500}}, {i4, {0, 4}}, {b, {0, 8}}, {w, {0, 65}}},
        {{k, {0, 512}}}, subA1 * subB1);
    cout << range->toReadable() << endl;
    // doubleStages
    auto subA2 = makeSubscript(A, {b, i45, k});
    auto subB2 = makeSubscript(B, {b, (i45 + w), k});
    auto innerRange =
        makeRangeOperator({{i45, {0, 10000}}, {b, {0, 8}}, {w, {0, 65}}},
                          {{k, {0, 512}}}, subA2 * subB2);
    auto subOuter = makeSubscript(innerRange, {(i3 + (2500 * i4)), b, w});
    auto outerRange = makeRangeOperator(
        {{i3, {0, 2500}}, {i4, {0, 4}}, {b, {0, 8}}, {w, {0, 65}}}, {},
        subOuter);
    cout << outerRange->toReadable() << endl;

    auto dataA = make_ref<vector<int>>(8 * 10000 * 512);
    auto dataB = make_ref<vector<int>>(8 * 10000 * 512);
    for (int i = 0; i < 8 * 10000 * 512; i++) {
        dataA->operator[](i) = i;
        dataB->operator[](i) = i;
    }
    unordered_map<string, Ref<vector<int>>> inputs{{"A", dataA}, {"B", dataB}};
    vector<vector<int>> positions{{0, 0, 0, 0}, {1, 2, 3, 4}};
    auto values1 = Interpreter(inputs).interpret(range, positions);
    auto values2 = Interpreter(inputs).interpret(outerRange, positions);

    EXPECT_EQ(values1, values2);
}

// L<n:0:1><h:0:4><w:0:4><c:0:256>Sum  ...  [n,c,((h + 1) / 2),((h + 1) % 2),((w
// + 1) / 2),((w + 1) % 2)]
// {L<n:0:1><c:0:256><x1:0:3><x2:0:2><y1:0:3><y2:0:2>Sum<f:0:448><r:0:2><s:0:2>
// {({A<pad=0,2,2,0>}[n, ((x1 + r) + -1), ((y1 + s) + -1), f] * {K}[((2 - (2 *
// r)) + x2), ((2 - (2 * s)) + y2), f, c])}}
// ==> A : Input Tensor shape=[1, 4, 4, 448] pad=[0, 2, 2, 0]
// ==> K : Input Tensor shape=[2, 2, 448, 256] pad=[0, 0, 0, 0]
TEST(Interpreter, TransConv) {
    DEFINE_VAR(n);
    DEFINE_VAR(h);
    DEFINE_VAR(w);
    DEFINE_VAR(c);
    DEFINE_VAR(x1);
    DEFINE_VAR(x2);
    DEFINE_VAR(y1);
    DEFINE_VAR(y2);
    DEFINE_VAR(f);
    DEFINE_VAR(r);
    DEFINE_VAR(s);
    auto A = makeTensor("A", {1, 4, 4, 448}, {0, 2, 2, 0});
    auto K = makeTensor("K", {4, 4, 448, 256}, {0, 0, 0, 0});
    auto subA = makeSubscript(A, {n, ((x1 + r) + (-1)), ((y1 + s) + (-1)), f});
    auto subK =
        makeSubscript(K, {((2 - (2 * r)) + x2), ((2 - (2 * s)) + y2), f, c});
    auto innerRange = makeRangeOperator(
        {{n, {0, 1}},
         {c, {0, 256}},
         {x1, {0, 3}},
         {x2, {0, 2}},
         {y1, {0, 3}},
         {y2, {0, 2}}},
        {{f, {0, 448}}, {r, {0, 2}}, {s, {0, 2}}}, subA * subK);
    auto subOuter =
        makeSubscript(innerRange, {n, c, ((h + 1) / 2), ((h + 1) % 2),
                                   ((w + 1) / 2), ((w + 1) % 2)});
    auto outerRange = makeRangeOperator(
        {{n, {0, 1}}, {h, {0, 4}}, {w, {0, 4}}, {c, {0, 256}}}, {}, subOuter);
    cout << outerRange->toReadable() << endl;

    auto dataA = make_ref<vector<int>>(1 * 4 * 4 * 448);
    auto dataK = make_ref<vector<int>>(4 * 4 * 448 * 256);
    for (int i = 0; i < 1 * 4 * 4 * 448; i++) {
        dataA->operator[](i) = i;
    }
    for (int i = 0; i < 4 * 4 * 448 * 256; i++) {
        dataK->operator[](i) = i;
    }
    unordered_map<string, Ref<vector<int>>> inputs{{"A", dataA}, {"K", dataK}};
    vector<vector<int>> positions{{0, 2, 2, 85}};
    auto vals = Interpreter(inputs).interpret(outerRange, positions);
    dbg(vals[0]);
}
