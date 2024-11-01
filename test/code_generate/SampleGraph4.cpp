#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include "test.h"

TEST(SAMPLE_GRAPH_4, Cuda_codeGenerate) {
    // conv7x7--->conv3x3--->add
    //         \->conv3x3-/
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, 3, 224, 224});
    auto i1 = g->tensor({1, 64, 56, 56});
    auto i2 = g->tensor({1, 128, 28, 28});
    auto i3 = g->tensor({1, 128, 28, 28});
    auto i4 = g->tensor({1, 128, 28, 28});

    auto w1 = g->tensor({64, 3, 7, 7});
    auto w2 = g->tensor({128, 64, 3, 3});
    auto w3 = g->tensor({128, 64, 3, 3});

    auto op1 = g->conv(i0, w1, i1, 3, 3, 4, 4);
    auto op2 = g->conv(i1, w2, i2, 1, 1, 2, 2);
    auto op3 = g->conv(i1, w3, i3, 1, 1, 2, 2);
    auto op4 = g->add({i2, i3}, i4);

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
    tmp = op2->compute();
    assert(tmp != nullptr);
    tmp = op3->compute();
    assert(tmp != nullptr);
    tmp = op4->compute();
    assert(tmp != nullptr);
    tpm::VType truth = i4->getData({0, 1, 2, 3});

    std::cout << "Get " << result << ", Expect " << truth << std::endl;
    assert(result == truth);

    delete g;
    delete h;
}
