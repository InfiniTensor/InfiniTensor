#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

TEST(SAMPLE_GRAPH_3, Cuda_codeGenerate) {
    // conv7x7->conv3x3->conv3x3
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, 3, 224, 224});
    auto i2 = g->tensor({1, 64, 56, 56});
    auto i4 = g->tensor({1, 128, 28, 28});
    auto i6 = g->tensor({1, 128, 28, 28});

    auto w1 = g->tensor({64, 3, 7, 7});
    auto w3 = g->tensor({128, 64, 3, 3});
    auto w5 = g->tensor({128, 128, 3, 3});

    auto op1 = g->conv(i0, w1, i2, 3, 3, 4, 4);
    auto op3 = g->conv(i2, w3, i4, 1, 1, 2, 2);
    auto op5 = g->conv(i4, w5, i6, 1, 1, 1, 1);

    g->updateConnection();

    int seed = 1234;

    auto h = new tpm::SubGraph(g->getOperators());
    for (auto &&t : g->getInputs()) {
        t->dataRand(seed);
    }
    for (auto &&t : h->getInputs()) {
        t->dataRand(seed);
    }

    std::cout << "computing partial" << std::endl;
    bool success;
    tpm::VType result;
    std::tie(success, result) = h->compute({0, 1, 2, 3});
    assert(success);

    std::cout << "computing full" << std::endl;
    tpm::Tensor *tmp;
    tmp = op1->compute();
    assert(tmp != nullptr);
    tmp = op3->compute();
    assert(tmp != nullptr);
    tmp = op5->compute();
    assert(tmp != nullptr);
    tpm::VType truth = i6->getData({0, 1, 2, 3});

    std::cout << "Get " << result << ", Expect " << truth << std::endl;
    assert(result == truth);

    delete g;
    delete h;
}
