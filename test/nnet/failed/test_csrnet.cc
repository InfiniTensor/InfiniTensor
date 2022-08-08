#include "code_engine.h"
#include "graph.h"
#include "nnet/derivator.h"
#include "nnet/dmutator.h"
#include "nnet/expr.h"
#include "nnet/visitor.h"
#include "operator.h"
#include "search_engine.h"
#include "gtest/gtest.h"
using namespace nnet;
using namespace std;
#define DEFINE_VAR(name) auto name = make_ref<VarNode>(#name);

void runCsrnetOpt(int batchSize) {
    const int N = 2 * batchSize, C = 512, H = 14, W = 14 / 2, R = 3, S = 3;
    auto g = new tpm::Graph();

    auto i0 = g->tensor({N, C, H, W});
    vector<tpm::Tensor *> w{
        g->tensor({512, 512, R, S}), g->tensor({512, 512, R, S}),
        g->tensor({512, 512, R, S}), g->tensor({256, 512, R, S}),
        g->tensor({128, 256, R, S}), g->tensor({64, 128, R, S})};

    const int nLayers = 6;
    i0 = g->transpose(i0, 2, {{0, -1}, 1, 2, 3}, 2)->getOutput();
    for (int i = 0; i < nLayers; ++i) {
        auto conv = g->conv(i0, w[i], 1, 1, 1, 1, 1, 1);
        auto relu = g->relu(conv->getOutput());
        i0 = relu->getOutput();
    }
    auto i1 = g->transpose(i0, 0, {0, 1, {2, -1}, 3}, 2)->getOutput();
    auto outputShape = i1->getDims();
    ASSERT_TRUE(outputShape[0] == N * 2);
    ASSERT_TRUE(outputShape[1] == 64);
    ASSERT_TRUE(outputShape[2] == H);
    ASSERT_TRUE(outputShape[3] == W / 2);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(make_shared<tpm::DMutator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    codeEngine.genCode(bestGraph, "res.cu");
}

TEST(CSRNET, Original) {
    const int N = 1, C = 512, H = 14, W = 14, R = 3, S = 3;
    auto g = new tpm::Graph();

    auto i0 = g->tensor({N, C, H, W});
    vector<tpm::Tensor *> w{
        g->tensor({512, 512, R, S}), g->tensor({512, 512, R, S}),
        g->tensor({512, 512, R, S}), g->tensor({256, 512, R, S}),
        g->tensor({128, 256, R, S}), g->tensor({64, 128, R, S})};

    const int nLayers = 6;
    for (int i = 0; i < nLayers; ++i) {
        auto conv = g->conv(i0, w[i], 2, 2, 1, 1, 2, 2);
        auto relu = g->relu(conv->getOutput());
        i0 = relu->getOutput();
    }
    auto outputShape = i0->getDims();
    ASSERT_TRUE(outputShape[0] == N);
    ASSERT_TRUE(outputShape[1] == 64);
    ASSERT_TRUE(outputShape[2] == H);
    ASSERT_TRUE(outputShape[3] == W);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(make_shared<tpm::DMutator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    codeEngine.genCode(bestGraph, "res.cu");
}

TEST(CSRNET, Optimized_BS1) { runCsrnetOpt(1); }
TEST(CSRNET, Optimized_BS16) { runCsrnetOpt(16); }