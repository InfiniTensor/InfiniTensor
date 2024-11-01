#include "code_gen/trans_eliminator.h"
#include "test.h"

TEST(ELIMINATOR_TEST_1, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    tpm::Dim dim = {4, 4, 14, 14};
    auto t1 = g->tensor(dim);
    auto t2 = g->tensor(dim);
    auto t3 = g->tensor(dim);
    auto t4 = g->tensor(dim);
    auto t5 = g->tensor(dim);
    auto t6 = g->tensor(dim);
    g->transpose(t1, t2, 2, {0, 1, {-1, 2}, 3}, 2);
    g->transpose(t2, t3, 0, {0, 1, 2, {-1, 3}}, 2);
    g->transpose(t3, t4, 0, {0, 1, 2, {-1, 3}}, 2);
    g->transpose(t4, t5, 3, {{0, 3}, 1, 2, -1}, -2);
    g->transpose(t5, t6, 3, {{0, 3}, 1, 2, -1}, -2);

    auto sg = std::make_shared<tpm::SubGraph>(g->getOperators());
    auto te = tpm::TransEliminator();
    std::cout << "before: " << std::endl;
    sg->print();
    auto ret = te.eliminate(sg);
    std::cout << "after: " << std::endl;
    if (ret == nullptr)
        std::cout << "empty" << std::endl;
    else
        ret->print();
}
