#include "code_gen/code_engine.h"
#include "code_gen/nnet/nmutator.h"
#include "code_gen/perf_engine.h"
#include "code_gen/search_engine.h"
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <onnx-file>" << std::endl;
        return -1;
    }
    auto g = new tpm::Graph();
    g->importOnnx(argv[1]);
    std::cout << "Graph Imported" << std::endl;

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::NMutator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");

    perfEngine->setPenalty(0);
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(graph, "origin.cu");

    return 0;
}
