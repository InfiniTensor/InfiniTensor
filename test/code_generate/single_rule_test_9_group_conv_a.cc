#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>
#include "test.h"

const int f = 16;
const int n = 1, c = 256, h = 28, w = 28;
const int f0 = 2 * f, f1 = 3 * f, f2 = 2 * f, f3 = 4 * f, r = 3, s = 3;

TEST(SINGLE_RULE_TEST_9_CONV_A, Cuda_codeGenerate) {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({n, c, h, w});
    // auto i1 = g->tensor({n, c, h, w});
    // auto i2 = g->tensor({n, c, h, w});
    auto i3 = g->tensor({n, c, h, w});
    auto w0 = g->tensor({f0, c, r, s});
    auto w1 = g->tensor({f1, c, r, s});
    auto w2 = g->tensor({f2, c, r, s});
    auto w3 = g->tensor({f3, c, r, s});
    auto o0 = g->tensor({n, f0, h, w});
    auto o1 = g->tensor({n, f1, h, w});
    auto o2 = g->tensor({n, f2, h, w});
    auto o3 = g->tensor({n, f3, h, w});
    auto op0 = g->conv(i0, w0, o0, 1, 1, 1, 1, 1, 1);
    auto op1 = g->conv(i0, w1, o1, 1, 1, 1, 1, 1, 1);
    auto op2 = g->conv(i0, w2, o2, 1, 1, 1, 1, 1, 1);
    auto op3 = g->conv(i3, w3, o3, 1, 1, 1, 1, 1, 1);

    auto sg = new tpm::SubGraph({op0, op1, op2, op3});
    for (auto tensor : sg->getTensors())
        tensor->dataMalloc();
    // for (auto tensor : sg->getInputs())
    //     tensor->dataRand();
    // for (auto op : sg->getOperators())
    //     op->compute();

    std::vector<std::shared_ptr<tpm::Operator>> all_ops;
    // all_ops.emplace_back(new tpm::ExtendOp(1, 1));
    // all_ops.emplace_back(new tpm::ExtendOp(1, 2));
    // all_ops.emplace_back(new tpm::ConcatOp(0));
    // all_ops.emplace_back(new tpm::ConcatOp(1));
    // all_ops.emplace_back(new tpm::ConvOp(1, 1, 1, 1, 1, 1));
    // all_ops.emplace_back(new tpm::SplitOp(1, std::vector<int>{2, 3}));
    tpm::Generator mutant{};
    std::vector<tpm::SubGraph *> candidates;
    // mutant.run(sg, candidates, 6, all_ops);
    mutant.run(sg, candidates);
    std::cout << "candidates found: " << candidates.size() << std::endl;
    for (auto candidate : candidates)
        candidate->print();

    assert(candidates.size() == 2);

    delete g;
    delete sg;
}
