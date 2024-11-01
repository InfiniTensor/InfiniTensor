#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int m = 8, n = 8, k = 4;

TEST(SINGLE_RULE_TEST_7_TRANS_BGEMM, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, m, k});
    auto w0 = g->tensor({1, k, n});
    auto i1 = g->tensor({1, m, n});

    auto op0 = g->matmul(i0, w0, i1);

    auto sg = new tpm::SubGraph({op0});
    for (auto tensor : sg->getTensors())
        tensor->dataMalloc();
    // for (auto tensor : sg->getInputs())
    //     tensor->dataRand();
    // for (auto op : sg->getOperators())
    //     op->compute();

    std::vector<std::shared_ptr<tpm::Operator>> all_ops;
    all_ops.emplace_back(new tpm::MatmulOp(true, false));
    all_ops.emplace_back(new tpm::TransposeOp(-1, {0, 2, 1}));
    tpm::Generator mutant{};
    std::vector<tpm::SubGraph *> candidates;
    mutant.run(sg, candidates, 2, all_ops);
    std::cout << "candidates found: " << candidates.size() << std::endl;
    for (auto candidate : candidates) {
        candidate->print();
    }

    assert(candidates.size() == 1);

    delete g;
    delete sg;
}
