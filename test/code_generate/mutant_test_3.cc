#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

TEST(MUTANT_TEST_3, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({1, 1, 14, 14});
    auto i1 = g->tensor({1, 1, 14, 14});
    auto op0 = g->identity(i0, i1);

    auto sg = new tpm::SubGraph({op0});
    for (auto tensor : sg->getTensors())
        tensor->dataMalloc();
    for (auto tensor : sg->getInputs())
        tensor->dataRand();
    for (auto op : sg->getOperators())
        op->compute();

    std::vector<std::shared_ptr<tpm::Operator>> all_ops;
    all_ops.emplace_back(new tpm::TransposeOp(2, {0, 1, {-1, 2}, 3}, -2));
    all_ops.emplace_back(new tpm::TransposeOp(2, {0, 1, {-1, 2}, 3}, 2));
    // all_ops.emplace_back(new tpm::TransposeOp(3, {0, 1, 2, {-1, 3}}, -2));
    // all_ops.emplace_back(new tpm::TransposeOp(2, {0, {1, -1}, 2, 3}, -2));
    // all_ops.emplace_back(new tpm::TransposeOp(2, {0, 1, {-1, 2}, 3}, -2));
    // all_ops.emplace_back(new tpm::TransposeOp(2, {0, {-1, 1}, 2, 3}, -2));
    // all_ops.emplace_back(new tpm::TransposeOp(0, {0, 1, {-1, 2}, 3}));
    // all_ops.emplace_back(new tpm::TransposeOp(2, {(0, -1), 1, 2, 3}));
    // all_ops.emplace_back(new tpm::TransposeOp(0, {0, 1, 2, {-1, 3}}));
    // all_ops.emplace_back(new tpm::ConvOp(1, 1, 1, 1, 1, 1));
    tpm::Generator mutant{false}; // disable prune_reciprocity
    std::vector<tpm::SubGraph *> candidates;
    mutant.run(sg, candidates, 2, all_ops, 0.99F);
    std::cout << "candidates found: " << candidates.size() << std::endl;
    for (auto candidate : candidates)
        candidate->print();

    delete g;
    delete sg;
}
