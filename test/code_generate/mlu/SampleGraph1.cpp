#include "code_gen/mlu/code_engine_mlu.h"
#include "code_gen/mlu/graph.h"
#include "code_gen/mlu/operator.h"
#include "code_gen/mlu/search_engine.h"
#include "code_gen/mlu/tensor.h"
#include "test.h"

TEST(SAMPLE_GRAPH_1, Cuda_codeGenerate) {
    // conv7x7->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu->conv3x3->relu
    auto g = new tpm::Graph();
    auto i0 = g->tensor({16, 3, 224, 224});
    auto i1 = g->tensor({16, 64, 56, 56});
    auto i2 = g->tensor({16, 64, 56, 56});

    auto w1 = g->tensor({64, 3, 7, 7});

    g->conv(i0, w1, i1, 3, 3, 4, 4);
    g->relu(i1, i2);

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
