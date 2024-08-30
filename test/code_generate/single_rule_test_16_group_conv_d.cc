#include "code_gen/generator.h"
#include "code_gen/graph.h"
#include "code_gen/operator.h"
#include "code_gen/search_engine.h"
#include "code_gen/tensor.h"
#include <cstdlib>
#include <iostream>

const int n = 1, c = 2, h = 14, w = 14;
const int f0 = 4, f1 = 6, r = 3, s = 3;

int main() {
    auto g = new tpm::Graph();
    auto i0 = g->tensor({n, c * 2, h, w});
    auto w0 = g->tensor({f0, c, r, s});
    auto w1 = g->tensor({f1, c, r, s});
    auto o0 = g->tensor({n, f0, h, w});
    auto o1 = g->tensor({n, f1, h, w});
    // auto o2 = g->tensor({n, f0 + f1, h, w});
    auto op0 = g->conv(i0, w0, o0, 1, 1, 1, 1, 1, 1);
    auto op1 = g->conv(i0, w1, o1, 1, 1, 1, 1, 1, 1);
    // auto op2 = g->concat({o0, o1}, o2, 1);


    auto sg = new tpm::SubGraph({op0, op1});
    for (auto tensor : sg->getTensors())
        tensor->dataMalloc();
    // for (auto tensor : sg->getInputs())
    //     tensor->dataRand();
    // for (auto op : sg->getOperators())
    //     op->compute();

    std::vector<std::shared_ptr<tpm::Operator>> all_ops;
    all_ops.emplace_back(new tpm::ConcatOp(0));
    all_ops.emplace_back(new tpm::ConcatOp(1));
    all_ops.emplace_back(new tpm::ConvOp(1, 1, 1, 1, 1, 1));
    // all_ops.emplace_back(new tpm::SplitOp(1, std::vector<int>{2, 3}));
    tpm::Generator mutant{};
    std::vector<tpm::SubGraph *> candidates;
    mutant.run(sg, candidates, 3, all_ops);
    std::cout << "candidates found: " << candidates.size() << std::endl;
    for (auto candidate : candidates)
        candidate->print();

    // assert(candidates.size() == 3);

    delete sg;
    delete g;
    return 0;
}
