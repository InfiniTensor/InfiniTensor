#include "code_gen/cmutator.h"
#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/perf_engine.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

TEST(CACHE_TEST_2, Cuda_codeGenerate) {
    // conv7x7->relu->conv3x3->relu
    auto g = new tpm::Graph();
    auto i0 = g->tensor({16, 3, 224, 224});
    auto i1 = g->tensor({16, 64, 56, 56});
    auto i2 = g->tensor({16, 64, 56, 56});
    auto i3 = g->tensor({16, 128, 28, 28});
    auto i4 = g->tensor({16, 128, 28, 28});

    auto w1 = g->tensor({64, 3, 7, 7});
    auto w3 = g->tensor({128, 64, 3, 3});

    g->conv(i0, w1, i1, 3, 3, 4, 4);
    g->relu(i1, i2);
    g->conv(i2, w3, i3, 1, 1, 2, 2);
    g->relu(i3, i4);

    g->updateConnection();
    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::CMutator>());
    searchEngine.run(graph, bestGraph);
}