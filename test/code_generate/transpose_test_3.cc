#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int n = 2, c = 1, h = 6, w = 6;
const int f = 1, r = 3, s = 3;

TEST(TRNASPOSE_TEST_3, Cuda_codeGenerate) {
    auto i0 = new tpm::Tensor({n, c, h, w});
    auto trans0 = new tpm::TransposeOp(i0, 0, {0, 1, {-1, 2}, 3}, 2);
    i0->dataMalloc();
    auto i0d = i0->getDataPtr();
    for (size_t i = 0; i < i0->size(); ++i)
        i0d[i] = i;
    std::cout << "input: " << std::endl;
    i0->print();

    auto i1 = trans0->compute();
    std::cout << "output 1: " << std::endl;
    i1->print();

    auto trans1 = new tpm::TransposeOp(i1, 2, {{0, 2}, 1, -1, 3}, -2);
    auto i2 = trans1->compute();
    std::cout << "output 2: " << std::endl;
    i2->print();

    delete trans0;
    delete i0;
    delete i1;
    delete i2;
}
