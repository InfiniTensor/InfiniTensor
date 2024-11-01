#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int n = 6;

TEST(TRNASPOSE_TEST_2, Cuda_codeGenerate) {
    auto i0 = new tpm::Tensor({1, 1, 6, 6});
    auto i1 = new tpm::Tensor({1, 1, 6, 6});
    auto i2 = new tpm::Tensor({1, 1, 6, 6});
    auto trans0 = new tpm::TransposeOp(i0, i1, 2, {0, 1, {-1, 2}, 3}, 2);
    auto trans1 = new tpm::TransposeOp(i1, i2, 3, {0, 1, 2, {-1, 3}}, 2);
    i0->dataMalloc();
    i1->dataMalloc();
    i2->dataMalloc();
    auto i0d = i0->getDataPtr();
    auto i1d = i1->getDataPtr();
    auto i2d = i2->getDataPtr();
    for (size_t i = 0; i < i0->size(); ++i)
        i0d[i] = i;
    std::cout << "input: " << std::endl;
    for (size_t i = 0; i < i0->size(); ++i) {
        std::cout << i0d[i] << ", ";
        if (i % n == n - 1)
            std::cout << std::endl;
    }

    trans0->compute();

    std::cout << "output: " << std::endl;
    for (size_t i = 0; i < i1->size(); ++i) {
        std::cout << i1d[i] << ", ";
        if (i % n == n - 1)
            std::cout << std::endl;
    }

    trans1->compute();

    std::cout << "output: " << std::endl;
    for (size_t i = 0; i < i2->size(); ++i) {
        std::cout << i2d[i] << ", ";
        if (i % n == n - 1)
            std::cout << std::endl;
    }

    delete trans0;
    delete trans1;
    delete i0;
    delete i1;
    delete i2;
}
