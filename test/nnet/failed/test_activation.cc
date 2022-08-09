#include "code_engine.h"
#include "nnet/expr.h"
#include "nnet/nmutator.h"
#include "operator.h"
#include "search_engine.h"
#include "tensor.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

TEST(Activation, Relu) {
    const int n_heads = 8, seq_len = 10000, feat_len = 512;
    // dilation_heads = 2;
    const int Batch = n_heads, M = seq_len, K = feat_len, W = 32;
    DEFINE_VAR(b);
    DEFINE_VAR(m);
    DEFINE_VAR(w);
    DEFINE_VAR(k);
    auto A = make_ref<TensorNode>("A", vector<int>({Batch, M, K}),
                                  vector<int>{0, 0, 0});

    auto subA = makeSubscript(A, {b, m, k});
    auto innerRange = makeRangeOperator(
        {{b, {0, Batch}}, {m, {0, M}}, {k, {0, K}}}, {}, subA);
    auto outerSub = makeSubscript(innerRange, {b, m, k});
    // auto subB = makeSubscript(B, {b, m + dilation * (w - W), k});
    auto relu = make_ref<FuncNode>(subA, FuncType::Relu);
    auto range =
        makeRangeOperator({{b, {0, Batch}}, {m, {0, M}}, {w, {0, 2 * W + 1}}},
                          {{k, {0, K}}}, relu);
    dbg(range);

    auto g = new tpm::Graph();
    auto i0 = g->tensor({Batch, M, K});
    auto i1 = g->tensor({Batch, M, 2 * W + 1});

    tpm::TensorVec inputsT{i0};
    tpm::TensorVec outputsT{i1};
    g->membound(inputsT, outputsT, {A}, range, 0);

    g->updateConnection();
    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(make_shared<tpm::NMutator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    codeEngine.genCode(bestGraph, "res.cu");
}
