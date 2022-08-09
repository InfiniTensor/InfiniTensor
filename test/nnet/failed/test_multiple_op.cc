#include "nnet/nmutator.h"
#include "operator.h"
#include "search_engine.h"
#include "tensor.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <iostream>

const int m = 8, n = 8, k = 4;

TEST(MULTIPLE_OP, main) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, m, k});
    auto w0 = g->tensor({1, k, n});
    auto w1 = g->tensor({1, k, n});
    auto i1 = g->tensor({1, m, n});
    auto i2 = g->tensor({1, m, n});
    // auto i3 = g->tensor({1, m * 2, n});

    g->matmul(i0, w0, i1);
    g->matmul(i0, w1, i2);
    // auto op2 = g->concat({i1, i2}, i3, 1);

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g->getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::NMutator>());
    searchEngine.run(graph, bestGraph);

    delete g;
}
