#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int m = 8, n = 8, k = 4;

TEST(SINGLE_RULE_TEST_6_BGEMM, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, m, k});
    auto w0 = g->tensor({1, k, n});
    auto w1 = g->tensor({1, k, n});
    auto i1 = g->tensor({1, m, n});
    auto i2 = g->tensor({1, m, n});
    // auto i3 = g->tensor({1, m * 2, n});

    auto op0 = g->matmul(i0, w0, i1);
    auto op1 = g->matmul(i0, w1, i2);
    // auto op2 = g->concat({i1, i2}, i3, 1);

    auto sg = new tpm::SubGraph({op0, op1});
    for (auto tensor : sg->getTensors())
        tensor->dataMalloc();
    // for (auto tensor : sg->getInputs())
    //     tensor->dataRand();
    // for (auto op : sg->getOperators())
    //     op->compute();

    std::vector<std::shared_ptr<tpm::Operator>> all_ops;
    // all_ops.emplace_back(new tpm::ConcatOp(0));
    // all_ops.emplace_back(new tpm::MatmulOp(false, false));
    // all_ops.emplace_back(new tpm::SplitOp(0, {1, 1}));
    tpm::Generator mutant{};
    std::vector<tpm::SubGraph *> candidates;
    mutant.run(sg, candidates);
    std::cout << "candidates found: " << candidates.size() << std::endl;
    for (auto candidate : candidates) {
        candidate->print();
    }

    assert(candidates.size() == 1);

    delete g;
    delete sg;
}
