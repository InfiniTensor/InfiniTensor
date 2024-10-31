#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int n = 16, c = 16, h = 14, w = 14;
const int f = 32, r = 3, s = 3;

TEST(CONV_TEST_3, Cuda_codeGenerate) {
    auto i0 = new tpm::Tensor({n, c, h, w});
    i0->dataRand();
    auto i0d = i0->getDataPtr();
    for (size_t i = 0; i < i0->size(); ++i)
        i0d[i] = i;

    auto w0 = new tpm::Tensor({f, c, r, s});
    w0->dataRand();
    auto w0d = w0->getDataPtr();
    for (size_t i = 0; i < w0->size(); ++i)
        w0d[i] = i;

    auto o1 = new tpm::Tensor({n, f, h, w});
    auto o2 = new tpm::Tensor({n, f, h, w});
    o1->dataMalloc();
    o2->dataMalloc();
    auto o1d = o1->getDataPtr();
    auto o2d = o2->getDataPtr();
    auto conv1 = new tpm::ConvOp(i0, w0, o1, 1, 1, 1, 1, 1, 1);
    auto conv2 = new tpm::ConvOp(i0, w0, o2, 1, 1, 1, 1, 1, 1);

    conv1->compute();
    std::cout << "o1 compute finish" << std::endl;
    o2->itInit();
    while (o2->itValid()) {
        auto it = o2->itGet();
        // std::cout << "[";
        // for (auto i : it)
        //     std::cout << i << ", ";
        // std::cout << "]" << std::endl;
        conv2->compute(tpm::DimRange(it)).second();
        o2->itNext();
    }
    std::cout << "o2 compute finish" << std::endl;

    int total = 0, equal = 0;
    for (size_t i = 0; i < o1->size(); ++i) {
        total++;
        if (o1d[i] == o2d[i])
            equal++;
    }
    std::cout << "equal/total = " << equal << "/" << total << " = "
              << (double)equal / total << std::endl;
    assert(equal == total);

    delete conv1;
    delete conv2;
    delete i0;
    delete w0;
    delete o1;
    delete o2;
}
