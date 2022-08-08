#include "nnet/Visitor/MatchReshapeVisitor.h"
#include "nnet/expr.h"
#include "nnet/routine.h"
#include "nnet/test.h"
#include "gtest/gtest.h"
#include <chrono>
using namespace nnet;
using namespace std;

TEST(MatchReshape, ElementWise_NHWC) {
    DEFINE_VAR(i, c);
    auto A = make_ref<TensorNode>("A", vector<int>({1, 7, 7, 512}));
    auto subA = makeSubscript(A, {i / 49, i / 7, i % 7, c});
    auto expr = makeRangeOperator({{i, {0, 49}}, {c, {0, 512}}}, {}, subA);
    auto matchReshapeVisitor = MatchReshapeVisitor();
    EXPECT_TRUE(matchReshapeVisitor(expr));
}

TEST(MatchReshape, ElementWise_with_Sum) {
    DEFINE_VAR(n, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>{49, 512});
    auto subA = makeSubscript(
        A, {(49 * n + 7 * (h + r)) + (w + s), ((512 * r) + (512 * s)) + f});
    auto expr = makeRangeOperator(
        {{n, {0, 1}}, {h, {0, 7}}, {w, {0, 7}}, {f, {0, 512}}},
        {{r, {0, 1}}, {s, {0, 1}}}, subA);
    auto matchReshapeVisitor = MatchReshapeVisitor();
    EXPECT_TRUE(matchReshapeVisitor(expr));
}

// clang-format off
// One candiate of TEST(Conv2gemm1x1, NCHW_FCRS_search)
// ==> ROOT
// T26
// ==> T26 : EleWise{T15, }
// L<n:0:1><f:0:512><h:0:7><w:0:7>Sum<r:0:1><s:0:1>  ...  [((f + r) + s),(((49 * n) + (7 * (h + r))) + (w + s))]
//     {T15}
// ==> T15 : Matmul{bmnk = 1, 512, 49, 512; AB = T3, T14; transAB = 0, 0}
// L<transA:0:101><transB:0:100><swapAB:0:101>Sum  ...  [transA,transB]
//     {L<i39:0:49><i38:0:512>Sum<c:0:512>
//     {({T4}[i39, c] * {T3}[i38, c])}}
// ==> T3 : EleWise{K, }
// L<i38:0:512><c:0:512>Sum  ...  [i38,c,(i38 % 1),(i38 % 1)]
//     {K}
// ==> T14 : EleWise{A, }
// L<c:0:512><i39:0:49>Sum  ...  [i39,c]
//     {L<i39:0:49><c:0:512>Sum  ...  [(i39 / 49),c,(i39 / 7),(i39 % 7)]
//     {A}}
// ==> K : Input Tensor shape=[512,512,1,1] pad=[0,0,0,0]
// ==> A : Input Tensor shape=[1,512,7,7] pad=[0,0,0,0]
// clang-format on

TEST(MatchReshape, Conv2gemm_1x1_NCHW_K) {
    // ==> T3 : EleWise{K, }
    // L<i38:0:512><c:0:512>Sum  ...  [i38,c,(i38 % 1),(i38 % 1)]
    //     {K}
    // ==> K : Input Tensor shape=[512,512,1,1] pad=[0,0,0,0]
    DEFINE_VAR(i, c);
    auto A = make_ref<TensorNode>("K", vector<int>({512, 512, 1, 1}));
    auto subA = makeSubscript(A, {i, c, i % 1, i % 1});
    auto expr = makeRangeOperator({{i, {0, 512}}, {c, {0, 512}}}, {}, subA);
    auto matchReshapeVisitor = MatchReshapeVisitor();
    EXPECT_TRUE(matchReshapeVisitor(expr));
}

TEST(MatchReshape, Conv2gemm_1x1_NCHW_A_merged) {
    // ==> T6 : EleWise{A, }
    // L<c:0:512><i39:0:49>Sum  ...  [i39,c]
    //     {L<i39:0:49><c:0:512>Sum  ...  [(i39 / 49),c,(i39 / 7),(i39 % 7)]
    //     {A}}
    // ==> A : Input Tensor shape=[1,512,7,7] pad=[0,0,0,0]
    DEFINE_VAR(i, c);
    auto A = make_ref<TensorNode>("A", vector<int>({1, 512, 7, 7}));
    auto subA = makeSubscript(A, {(i / 49), c, (i / 7), (i % 7)});
    auto expr = makeRangeOperator({{c, {0, 512}}, {i, {0, 49}}}, {}, subA);
    auto matchReshapeVisitor = MatchReshapeVisitor();
    EXPECT_TRUE(matchReshapeVisitor(expr));
}

TEST(MatchReshape, Conv2gemm_1x1_NCHW_A) {
    // ==> T14 : EleWise{A, }
    // L<c:0:512><i39:0:49>Sum  ...  [i39,c]
    //     {L<i39:0:49><c:0:512>Sum  ...  [(i39 / 49),c,(i39 / 7),(i39 % 7)]
    //     {A}}
    // ==> A : Input Tensor shape=[1,512,7,7] pad=[0,0,0,0]
    DEFINE_VAR(i, c);
    auto A = make_ref<TensorNode>("A", vector<int>({1, 512, 7, 7}));
    auto subA = makeSubscript(A, {(i / 49), c, (i / 7), (i % 7)});
    auto inner = makeRangeOperator({{i, {0, 49}}, {c, {0, 512}}}, {}, subA);
    auto subInner = makeSubscript(inner, {i, c});
    auto outer = makeRangeOperator({{c, {0, 512}}, {i, {0, 49}}}, {}, subInner);
    EXPECT_TRUE(MatchReshapeVisitor()(outer));
}

TEST(MatchReshape, Conv2gemm_1x1_NCHW_Output) {
    // ==> T26 : EleWise{T15, }
    // L<n:0:1><f:0:512><h:0:7><w:0:7>Sum<r:0:1><s:0:1>  ...  [((f + r) +
    // s),(((49 * n) + (7 * (h + r))) + (w + s))] {T15}
    // ==> T15 : Matmul{bmnk = 1, 512, 49, 512; AB = T3, T14; transAB = 0, 0}
    DEFINE_VAR(n, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({512, 49}));
    auto subA = makeSubscript(
        A, {((f + r) + s), (((49 * n) + (7 * (h + r))) + (w + s))});
    auto expr = makeRangeOperator(
        {{n, {0, 1}}, {f, {0, 512}}, {h, {0, 7}}, {w, {0, 7}}},
        {{r, {0, 1}}, {s, {0, 1}}}, subA);
    auto matchReshapeVisitor = MatchReshapeVisitor();
    EXPECT_TRUE(matchReshapeVisitor(expr));
}

TEST(MatchReshape, Conv2gemm_1x1_NCHW_Output_wrong) {
    // ==> T22 : EleWise{T7, }
    // L<n:0:1><h:0:7><w:0:7><f:0:512>Sum<r:0:1><s:0:1>  ...  [(((49 * n) + (7 *
    // (h + r))) + (w + s)),((f + r) + s)]
    //     {T7}
    // ==> T7 : Matmul{bmnk = 1, 49, 512, 512; AB = T6, T3; transAB = 1, 1}
    DEFINE_VAR(n, h, w, f, r, s);
    auto A = make_ref<TensorNode>("A", vector<int>({49, 512}));
    auto subA = makeSubscript(
        A, {(((49 * n) + (6 * (h + r))) + (w + s)), ((f + r) + s)});
    auto expr = makeRangeOperator(
        {{n, {0, 1}}, {h, {0, 7}}, {w, {0, 7}}, {f, {0, 512}}},
        {{r, {0, 1}}, {s, {0, 1}}}, subA);
    auto matchReshapeVisitor = MatchReshapeVisitor();
    EXPECT_FALSE(matchReshapeVisitor(expr));
}

// MemBound[124644277](i0=0, o0=119, exec_time=0.0037384, NNet
// Inputs=[A<pad=0,0,0,3>,]) L<c:0:2048><i35:0:49>Sum  ...  [i35,c]
//     {L<i35:0:49><c:0:2048>Sum  ...  [(i35 / 49),c,(i35 / 7),(i35 % 7)]
//     {A<pad=0,0,0,3>}}

TEST(MatchReshape, Conv2gemm_1x7_A) {
    //     MemBound[124644277](i0=0, o0=119, exec_time=0.0037384, NNet
    //     Inputs=[A<pad=0,0,0,3>,])
    // L<c:0:2048><i35:0:49>Sum  ...  [i35,c]
    //     {L<i35:0:49><c:0:2048>Sum  ...  [(i35 / 49),c,(i35 / 7),(i35 % 7)]
    //     {A<pad=0,0,0,3>}}
    const int N = 1, C = 2048, H = 7, W = 7, R = 1, S = 7; // gcn_Conv_137
    DEFINE_VAR(i, c);
    auto A = make_ref<TensorNode>("A", vector<int>({N, C, H, W}),
                                  vector<int>{0, 0, R / 2, S / 2});
    auto subA = makeSubscript(A, {(i / 49), c, (i / 7), (i % 7)});
    auto inner = makeRangeOperator({{i, {0, 49}}, {c, {0, 2048}}}, {}, subA);
    auto subInner = makeSubscript(inner, {i, c});
    auto outer =
        makeRangeOperator({{c, {0, 2048}}, {i, {0, 49}}}, {}, subInner);
    dbg(outer);
    EXPECT_TRUE(MatchReshapeVisitor()(outer));
}