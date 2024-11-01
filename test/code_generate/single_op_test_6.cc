#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

const int n = 1, c = 2, h = 4, w = 4;
const int f0 = 4, f1 = 6, f2 = 4, f3 = 8, r = 3, s = 3;

TEST(SINGLE_OP_TEST_6, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({n, c, h, w});
    auto i1 = g->tensor({n, c, h, w});
    auto i2 = g->tensor({n, c, h, w});
    auto i3 = g->tensor({n, c, h, w});
    auto w0 = g->tensor({f0, c, r, s});
    auto w1 = g->tensor({f1, c, r, s});
    auto w2 = g->tensor({f2, c, r, s});
    auto w3 = g->tensor({f3, c, r, s});
    // auto o0 = g->tensor({n, f0, h, w});
    // auto o1 = g->tensor({n, f1, h, w});
    // auto o2 = g->tensor({n, f2, h, w});
    // auto o3 = g->tensor({n, f3, h, w});
    g->conv(i0, w0, 1, 1, 1, 1, 1, 1);
    g->conv(i1, w1, 1, 1, 1, 1, 1, 1);
    g->conv(i2, w2, 1, 1, 1, 1, 1, 1);
    g->conv(i3, w3, 1, 1, 1, 1, 1, 1);
    // auto op0 = g->conv(i0, w0, o0, 1, 1, 1, 1, 1, 1);
    // auto op1 = g->conv(i1, w1, o1, 1, 1, 1, 1, 1, 1);
    // auto op2 = g->conv(i2, w2, o2, 1, 1, 1, 1, 1, 1);
    // auto op3 = g->conv(i3, w3, o3, 1, 1, 1, 1, 1, 1);

    // auto sg = new tpm::SubGraph({op0, op1});
    // // auto sg = new tpm::SubGraph({op0, op1, op2, op3});
    // for (auto tensor : sg->getTensors())
    //     tensor->dataMalloc();
    // for (auto tensor : sg->getInputs())
    //     tensor->dataRand();
    // for (auto op : sg->getOperators())
    //     op->compute();

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
