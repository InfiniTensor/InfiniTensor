#include "code_gen/code_engine.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

const int m = 16, n = 1024, k = 1024;
using namespace tpm;

TEST(SINGLE_OP_TEST_7_BATCH_GEMM, Cuda_codeGenerate) {
    auto g = Graph{};
    auto i0 = g.tensor({m, k});
    auto w0 = g.tensor({k, n});
    auto w1 = g.tensor({k, n});
    auto w2 = g.tensor({k, n});
    // auto w3 = g.tensor({k, n});
    // auto w4 = g.tensor({k, n});
    g.matmul(i0, w0);
    g.matmul(i0, w1);
    g.matmul(i0, w2);

    g.updateConnection();

    std::shared_ptr<tpm::SubGraph> graph, bestGraph;
    graph = std::make_shared<tpm::SubGraph>(g.getOperators());
    tpm::SearchEngine searchEngine(std::make_shared<tpm::Generator>());
    searchEngine.run(graph, bestGraph);
    tpm::CodeEngine codeEngine;
    auto perfEngine = searchEngine.exportPerfEngine();
    codeEngine.importPerfEngine(perfEngine);
    codeEngine.genCode(bestGraph, "res.cu");
}
