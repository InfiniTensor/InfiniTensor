#include "code_gen/code_engine.h"
#include "code_gen/nnet/nmutator.h"
#include "code_gen/perf_engine.h"
#include "code_gen/search_engine.h"
#include <chrono>
#include <iostream>
using namespace std;
namespace ch {
using namespace std::chrono;
}
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <onnx-file>" << std::endl;
        return -1;
    }
    auto g = new tpm::Graph();
    g->importOnnx(argv[1]);
    std::cout << "Graph Imported" << std::endl;

    for (int i = 0; i <= 8; ++i) {
        ch::time_point<ch::high_resolution_clock, ch::nanoseconds> beg, end;
        beg = ch::high_resolution_clock::now();

        std::shared_ptr<tpm::SubGraph> graph, bestGraph;
        graph = std::make_shared<tpm::SubGraph>(g->getOperators());
        auto mutationEngine = std::make_shared<tpm::NMutator>();
        mutationEngine->setMaxDepth(i);
        tpm::SearchEngine searchEngine(mutationEngine);
        searchEngine.run(graph, bestGraph);
        printf("Best graph :\n");
        bestGraph->print();
        // tpm::CodeEngine codeEngine;
        // auto perfEngine = searchEngine.exportPerfEngine();
        // codeEngine.importPerfEngine(perfEngine);
        // codeEngine.genCode(bestGraph, "res.cu");

        // perfEngine->setPenalty(0);
        // codeEngine.importPerfEngine(perfEngine);
        // codeEngine.genCode(graph, "origin.cu");

        end = ch::high_resolution_clock::now();
        double t = ch::duration_cast<ch::duration<double>>(end - beg).count();
        // printf("====== maxdepth=%d \n", i);
        printf("Statistics: maxdepth %d , time %.3lf s, states %lld , "
               "candidate %lld\n",
               i, t, mutationEngine->cntStates, mutationEngine->cntCandidates);
    }

    return 0;
}
