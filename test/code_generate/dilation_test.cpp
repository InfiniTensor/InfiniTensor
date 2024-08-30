#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>

// conv(kernel size)-(number of filters)-(dilation rate)
//
// conv(3x3)-512-(dhxdw)
// relu
// conv(3x3)-512-(dhxdw)
// relu
// conv(3x3)-512-(dhxdw)
// relu
// conv(3x3)-256-(dhxdw)
// relu
// conv(3x3)-128-(dhxdw)
// relu
// conv(3x3)-64-(dhxdw)

int dilaConfig[4][6] = {{1, 1, 1, 1, 1, 1},
                        {2, 2, 2, 2, 2, 2},
                        {2, 2, 2, 4, 4, 4},
                        {4, 4, 4, 4, 4, 4}};

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./dilation_test config_id batch_size\n"
                  << "config_id is in {0, 1, 2, 3}\n"
                  << "Example: ./dilation_test 0 1\n";
    }
    size_t configId = strtol(argv[1], nullptr, 10);
    int n = strtol(argv[2], nullptr, 10);
    const int *dc = dilaConfig[configId];

    auto g = new tpm::Graph();
    auto i0 = g->tensor({n, 512, 14, 14});
    auto i1 = g->tensor({n, 512, 14, 14});
    auto i2 = g->tensor({n, 512, 14, 14});
    auto i3 = g->tensor({n, 512, 14, 14});
    auto i4 = g->tensor({n, 512, 14, 14});
    auto i5 = g->tensor({n, 512, 14, 14});
    auto i6 = g->tensor({n, 512, 14, 14});
    auto i7 = g->tensor({n, 256, 14, 14});
    auto i8 = g->tensor({n, 256, 14, 14});
    auto i9 = g->tensor({n, 128, 14, 14});
    auto i10 = g->tensor({n, 128, 14, 14});
    auto i11 = g->tensor({n, 64, 14, 14});
    auto i12 = g->tensor({n, 64, 14, 14});

    auto w1 = g->tensor({512, 512, 3, 3});
    auto w3 = g->tensor({512, 512, 3, 3});
    auto w5 = g->tensor({512, 512, 3, 3});
    auto w7 = g->tensor({256, 512, 3, 3});
    auto w9 = g->tensor({128, 256, 3, 3});
    auto w11 = g->tensor({64, 128, 3, 3});

    g->conv(i0, w1, i1, dc[0], dc[0], 1, 1, dc[0], dc[0]);
    g->relu(i1, i2);
    g->conv(i2, w3, i3, dc[1], dc[1], 1, 1, dc[1], dc[1]);
    g->relu(i3, i4);
    g->conv(i4, w5, i5, dc[2], dc[2], 1, 1, dc[2], dc[2]);
    g->relu(i5, i6);
    g->conv(i6, w7, i7, dc[3], dc[3], 1, 1, dc[3], dc[3]);
    g->relu(i7, i8);
    g->conv(i8, w9, i9, dc[4], dc[4], 1, 1, dc[4], dc[4]);
    g->relu(i9, i10);
    g->conv(i10, w11, i11, dc[5], dc[5], 1, 1, dc[5], dc[5]);
    g->relu(i11, i12);

    g->setInputs({i0});
    g->setOutputs({i12});
    g->updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::Generator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");

    return 0;
}
