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

const int n1 = 16, c1 = 256, h1 = 28, w1 = 28;
const int n2 = 16, c2 = 128, h2 = 14, w2 = 14;
const int f1 = 256, r1 = 3, s1 = 3;
const int f2 = 256, r2 = 3, s2 = 3;

TEST(CACHE_TEST_1, Cuda_codeGenerate) {
    std::cout << "Conv: (" << n1 << ", " << c1 << ", " << h1 << ", " << w1
              << ") * (" << f1 << ", " << c1 << ", " << r1 << ", " << s1 << ")"
              << std::endl;

    auto g = new tpm::Graph();
    auto i0 = g->tensor({n1, c1, h1, w1});
    auto w0 = g->tensor({f1, c1, r1, s1});
    auto o1 = g->tensor({n1, f1, h1, w1});
    auto op0 = g->conv(i0, w0, o1, 1, 1, 1, 1, 1, 1);

    auto sg0 = new tpm::SubGraph({op0});
    for (auto tensor : sg0->getTensors()) {
        tensor->dataMalloc();
    }
    for (auto tensor : sg0->getInputs()) {
        tensor->dataRand();
    }

    std::vector<tpm::SubGraph *> candidates0;
    tpm::CMutator generator0;
    generator0.run(sg0, candidates0);

    std::cout << "Conv: (" << n2 << ", " << c2 << ", " << h2 << ", " << w2
              << ") * (" << f2 << ", " << c2 << ", " << r2 << ", " << s2 << ")"
              << std::endl;

    auto i1 = g->tensor({n2, c2, h2, w2});
    auto w1 = g->tensor({f2, c2, r2, s2});
    auto o2 = g->tensor({n2, f2, h2, w2});
    auto op1 = g->conv(i1, w1, o2, 1, 1, 1, 1, 1, 1);

    auto sg1 = new tpm::SubGraph({op1});
    for (auto tensor : sg1->getTensors()) {
        tensor->dataMalloc();
    }
    for (auto tensor : sg1->getInputs()) {
        tensor->dataRand();
    }

    std::vector<tpm::SubGraph *> candidates1;
    tpm::CMutator generator1;
    generator1.run(sg1, candidates1);

    delete g;
    delete sg0;
    delete sg1;
}