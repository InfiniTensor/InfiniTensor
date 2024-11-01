#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

const int n = 1, c = 48, h = 38, w = 38;
const int f = 64, wc = 48, r = 5, s = 5;
TEST(SINGLE_OP_TEST_1, Cuda_codeGenerate) {
    // conv7x7->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu
    auto g = new tpm::Graph();
    auto i8 = g->tensor({n, c, h, w});

    auto w9 = g->tensor({f, wc, r, s});

    g->conv(i8, w9, 2, 2);

    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::Generator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");

}
