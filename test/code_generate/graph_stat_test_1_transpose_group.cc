#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/tensor.h"
#include <iostream>
#include "test.h"

const int n = 16, c = 16, h = 14, w = 14;
const int f = 32, r = 1, s = 7;

using namespace tpm;

TEST(CONCAT_TEST_1, Cuda_codeGenerate) {
    auto g = Graph{};
    auto i0 = g.tensor({n, c, h, w});
    auto i1 = g.tensor({n, c, h, w});
    auto w0 = g.tensor({f, c, r, s});
    auto w1 = g.tensor({f, c, s, r});
    auto op0 = g.conv(i0, w0, 0, 3);
    auto op1 = g.conv(i1, w1, 3, 0);

    auto sg = SubGraph({op0, op1});
    auto gen = Generator();
    std::cout << gen.statGraph(&sg) << std::endl;
}