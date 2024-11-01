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

TEST(MUTANT_TEST_2, Cuda_codeGenerate) {
    auto i0 = new tpm::Tensor({n, c, h, w});
    auto i1 = new tpm::Tensor({n, c, h, w});
    auto i2 = new tpm::Tensor({n, c, h, w});
    // auto i0 = new tpm::Tensor({1, 1, n, n});
    // auto i1 = new tpm::Tensor({1, 1, n, n});
    // auto i2 = new tpm::Tensor({1, 1, n, n});
    auto trans0 = new tpm::TransposeOp(i0, i1, 2, {0, 1, {-1, 2}, 3}, 2);
    auto trans1 = new tpm::TransposeOp(i1, i2, 3, {0, 1, 2, {-1, 3}}, 2);
    i0->dataRand();
    i1->dataMalloc();
    i2->dataMalloc();
    auto i0d = i0->getDataPtr();
    i1->getDataPtr();
    i2->getDataPtr();
    for (size_t i = 0; i < i0->size(); ++i)
        i0d[i] = i;
    // std::cout << "input: " << std::endl;
    // for (int i = 0; i < i0->size(); ++i) {
    //     std::cout << i0d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

    trans0->compute();

    // std::cout << "output: " << std::endl;
    // for (int i = 0; i < i1->size(); ++i) {
    //     std::cout << i1d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

    trans1->compute();

    // std::cout << "output: " << std::endl;
    // for (int i = 0; i < i2->size(); ++i) {
    //     std::cout << i2d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

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
    o2->getDataPtr();
    auto convd2 = new tpm::ConvOp(i0, w0, o1, 2, 2, 1, 1, 2, 2);
    auto convd1 = new tpm::ConvOp(i2, w0, o2, 1, 1, 1, 1, 1, 1);

    convd2->compute();
    convd1->compute();

    // std::cout << "o1: " << std::endl;
    // for (int i = 0; i < o1->size(); ++i) {
    //     std::cout << o1d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

    // std::cout << "o2: " << std::endl;
    // for (int i = 0; i < o2->size(); ++i) {
    //     std::cout << o2d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

    auto o3 = new tpm::Tensor({n, f, h, w});
    auto o4 = new tpm::Tensor({n, f, h, w});
    o3->dataMalloc();
    o4->dataMalloc();
    o3->getDataPtr();
    auto o4d = o4->getDataPtr();

    auto trans2 = new tpm::TransposeOp(o2, o3, 2, {0, 1, {-1, 2}, 3}, -2);
    auto trans3 = new tpm::TransposeOp(o3, o4, 3, {0, 1, 2, {-1, 3}}, -2);
    trans2->compute();
    trans3->compute();

    // std::cout << "o3: " << std::endl;
    // for (int i = 0; i < o3->size(); ++i) {
    //     std::cout << o3d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

    // std::cout << "o4: " << std::endl;
    // for (int i = 0; i < o4->size(); ++i) {
    //     std::cout << o4d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

    // std::cout << "o1: " << std::endl;
    // for (int i = 0; i < o1->size(); ++i) {
    //     std::cout << o1d[i] << ", ";
    //     if (i % n == n - 1)
    //         std::cout << std::endl;
    // }

    int total = 0, equal = 0;
    for (size_t i = 0; i < o1->size() && i < o4->size(); ++i) {
        total++;
        if (o1d[i] == o4d[i])
            equal++;
    }
    std::cout << "equal/total = " << equal << "/" << total << " = "
              << (double)equal / total << std::endl;

    delete trans0;
    delete trans1;
    delete trans2;
    delete trans3;
    delete convd1;
    delete convd2;
    delete i0;
    delete i1;
    delete i2;
    delete w0;
    delete o1;
    delete o2;
    delete o3;
    delete o4;
}
