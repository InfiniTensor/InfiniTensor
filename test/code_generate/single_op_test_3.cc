#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

TEST(SINGLE_OP_TEST_3, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto a = g->tensor({1, 2048, 2048});
    auto b = g->tensor({1, 2048, 2048});
    auto c = g->tensor({1, 2048, 2048});

    g->matmul(a, b, c);

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
