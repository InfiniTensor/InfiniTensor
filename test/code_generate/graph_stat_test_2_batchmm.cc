#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/tensor.h"
#include <iostream>
#include "test.h"

const int m = 8, n = 8, k = 4;

using namespace tpm;

TEST(CONCAT_TEST_1, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, m, k});
    auto w0 = g->tensor({1, k, n});
    auto w1 = g->tensor({1, k, n});
    auto i1 = g->tensor({1, m, n});
    auto i2 = g->tensor({1, m, n});

    auto w2 = g->tensor({1, k, n + 1});
    auto i3 = g->tensor({1, m, n + 1});

    auto op0 = g->matmul(i0, w0, i1);
    auto op1 = g->matmul(i0, w1, i2);
    auto op2 = g->matmul(i0, w2, i3);

    auto sg = SubGraph({op0, op1});
    auto gen = Generator();
    std::cout << gen.statGraph(&sg) << std::endl;

    sg = SubGraph({op0, op2});
    std::cout << gen.statGraph(&sg) << std::endl;
}