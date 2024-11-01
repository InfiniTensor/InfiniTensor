#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int n = 1, c = 2, h = 4, w = 4;
const int f0 = 4, f1 = 6, r = 3, s = 3;

TEST(GROUP_CONV_TEST_1, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({n, c, h, w});
    auto i1 = g->tensor({n, c, h, w});
    auto w0 = g->tensor({f0, c, r, s});
    auto w1 = g->tensor({f1, c, r, s});
    i0->dataRand();
    i1->dataRand();
    w0->dataRand();
    w1->dataRand();

    auto e0 = g->extend(i0, 1, 1);
    auto e1 = g->extend(i1, 1, 2);
    auto i2 = e0->compute();
    auto i3 = e1->compute();

    auto cc0 = g->concat({i2, i3}, 1);
    auto cc1 = g->concat({w0, w1}, 0);
    auto i4 = cc0->compute();
    auto w4 = cc1->compute();

    auto c0 = g->conv(i4, w4, 1, 1);
    auto i5 = c0->compute();

    auto s0 = dynamic_cast<tpm::SplitOp *>(g->split(i5, 1, {2, 3}));
    auto os = s0->computeV();
    auto o0 = os[0], o1 = os[1];
    std::cout << "o0:" << std::endl;
    o0->print();
    std::cout << "o1:" << std::endl;
    o1->print();

    auto conv0 = g->conv(i0, w0, 1, 1);
    auto conv1 = g->conv(i1, w1, 1, 1);
    auto tensor0 = conv0->compute();
    auto tensor1 = conv1->compute();
    std::cout << "tensor0:" << std::endl;
    tensor0->print();
    std::cout << "tensor1:" << std::endl;
    tensor1->print();

    int total = 0, equal = 0;
    for (size_t i = 0; i < o0->size(); ++i) {
        total++;
        if (o0->getData(i) == tensor0->getData(i))
            equal++;
    }
    for (size_t i = 0; i < o1->size(); ++i) {
        total++;
        if (o1->getData(i) == tensor1->getData(i))
            equal++;
    }
    std::cout << "equal/total = " << equal << "/" << total << std::endl;

    delete g;
}
