#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int m = 8, n = 8, k = 4;

TEST(SINGLE_RULE_TEST_5_DILATED, Cuda_codeGenerate) {
    auto g1 = tpm::Graph{};
    auto i0 = g1.tensor({1, m, k});
    auto w0 = g1.tensor({1, k, n});
    auto w1 = g1.tensor({1, k, n});
    auto i1 = g1.tensor({1, m, n});
    auto i2 = g1.tensor({1, m, n});
    auto op0 = g1.matmul(i0, w0, i1);
    auto op1 = g1.matmul(i0, w1, i2);
    g1.updateConnection();

    auto g2 = tpm::Graph{};
    auto l0 = g2.tensor({1, m, k});
    auto r0 = g2.tensor({1, k, n});
    auto r1 = g1.tensor({1, k, n});

    auto c0 = g2.concat({l0, l0}, 0);
    auto c1 = g2.concat({r0, r1}, 0);
    auto l2 = c0->getOutputs()[0];
    auto r2 = c1->getOutputs()[0];
    auto mm = g2.matmul(l2, r2);
    auto t3 = mm->getOutputs()[0];
    // auto sp = g2.split(t3, 0, {1, 1});
    g2.split(t3, 0, {1, 1});
    g2.updateConnection();

    for (auto t : g1.getTensors())
        t->dataMalloc();
    for (auto t : g2.getTensors())
        t->dataMalloc();

    auto i0ptr = i0->getDataPtr(), w0ptr = w0->getDataPtr(),
         w1ptr = w1->getDataPtr();
    for (size_t i = 0; i < i0->size(); ++i)
        i0ptr[i] = i;
    for (size_t i = 0; i < w0->size(); ++i)
        w0ptr[i] = i * 3;
    for (size_t i = 0; i < w1->size(); ++i)
        w1ptr[i] = i * 7;

    l0->setData(i0ptr);
    r0->setData(w0ptr);
    r1->setData(w1ptr);

    for (auto op : g1.getOperators())
        op->compute();

    // c0->compute();
    // c1->compute();
    // mm->compute();
    // dynamic_cast<tpm::SplitOp *>(sp)->computeV();

    auto o0 = op0->getOutputs()[0], o1 = op1->getOutputs()[0];
    // auto a0 = sp->getOutputs()[0], a1 = sp->getOutputs()[1];

    std::cout << "o0:" << std::endl;
    o0->print();
    std::cout << "o1:" << std::endl;
    o1->print();

    // std::cout << "a0:" << std::endl;
    // a0->print();
    // std::cout << "a1:" << std::endl;
    // a1->print();
    auto sg2 = tpm::SubGraph(g2.getOperators());
    for (auto t : sg2.getTensors())
        t->dataMalloc();
    sg2.getInputs()[0]->setData(i0ptr);
    sg2.getInputs()[1]->setData(w0ptr);
    sg2.getInputs()[2]->setData(w1ptr);
    std::cout << (sg2.compute({0, 0, 0}).second) << std::endl;
    std::cout << (sg2.compute({0, 1, 3}).second) << std::endl;
    std::cout << (sg2.compute({0, 2, 5}).second) << std::endl;
    std::cout << (sg2.compute({0, 2, 2}).second) << std::endl;
    std::cout << (sg2.compute({0, 4, 1}).second) << std::endl;
    std::cout << (sg2.compute({0, 6, 3}).second) << std::endl;
    std::cout << (sg2.compute({0, 4, 2}).second) << std::endl;
}
