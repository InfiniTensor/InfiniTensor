#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <iostream>
#include "test.h"

const int n = 2, c = 1, h = 8, w = 8;

TEST(TRNASPOSE_TEST_5, Cuda_codeGenerate) {
    auto i0 = new tpm::Tensor({n, c, h, w});
    // tpm::Perm perm = {0, 1, {-1, 2}, 3};
    // int split = 2, factor = 2;
    tpm::Perm perm = {0, 1, {-1, 2}, 3};
    int split = 0, factor = 2;
    auto trans1 = new tpm::TransposeOp(i0, split, perm, factor);
    auto o1 = trans1->getOutputs()[0];

    std::cout << "Input tensor size: " << tpm::dimToString(i0->getDims())
              << std::endl;
    std::cout << "Output tensor size: " << tpm::dimToString(o1->getDims())
              << std::endl;
    trans1->print();
    std::cout << std::endl;
    auto ttParam = trans1->getTTParam();
    for (auto param : ttParam)
        param->print();

    delete trans1;
    delete i0;
    delete o1;
}
