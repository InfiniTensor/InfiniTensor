#include "code_gen/cmutator.h"
#include "code_gen/generator.h"

using namespace tpm;

Reciprocity::Reciprocity(const std::vector<std::shared_ptr<Operator>> &ops) {
    search_reciprocity(ops);
}

void Reciprocity::search_reciprocity(
    const std::vector<std::shared_ptr<Operator>> &ops) {
    // construct a graph to find reciprocities
    auto g = new tpm::Graph();
    auto i0 = g->tensor({6, 6, 14, 14});
    auto i1 = g->tensor({6, 6, 14, 14});
    auto op0 = g->identity(i0, i1);
    auto sg = new tpm::SubGraph({op0});
    for (auto tensor : sg->getTensors())
        tensor->dataMalloc();
    for (auto tensor : sg->getInputs())
        tensor->dataRand();
    for (auto op : sg->getOperators())
        op->compute();
    std::vector<std::shared_ptr<Operator>> candidate_ops;
    for (auto &op : ops)
        if (op->getType() == Operator::OpType::Transpose)
            candidate_ops.emplace_back(op->clone());

    // limit the depth below 3 to avoid large overhead
    tpm::Generator mutant(false);
    std::vector<tpm::SubGraph *> candidates;
    mutant.run(sg, candidates, MAX_RECIPROCITY_DETECT_DEPTH, candidate_ops,
               0.99F);

    for (auto candidate : candidates) {
        std::vector<uint64_t> chain;
        for (auto op : candidate->getOperators()) {
            chain.push_back(op->getHash());
        }
        reciprocal_op_chains.emplace_back(chain);
    }

    // // debug output
    // for (auto &chain : reciprocal_op_chains) {
    //     printf("Reciprocity: ");
    //     for (auto v : chain)
    //         printf("%3lu -> ", v);
    //     puts("");
    // }
}

bool Reciprocity::is_tail_reciprocity(const OpVec &oplist) {
    std::vector<uint64_t> cur_chain;
    if (oplist.empty())
        return false;
    auto cur_op_it = oplist.back();
    for (int i = 0; i < MAX_RECIPROCITY_DETECT_DEPTH; ++i) {
        if (!cur_op_it || cur_op_it->getType() != Operator::OpType::Transpose)
            break;
        cur_chain.push_back(cur_op_it->getHash());
        cur_op_it = cur_op_it->getInputs()[0]->getOutputOf();
    }
    if (cur_chain.empty())
        return false;
    for (auto &target_chain : reciprocal_op_chains) {
        if (target_chain.size() > cur_chain.size())
            continue;
        bool matched = true;
        for (int i = 0; i < (int)target_chain.size(); ++i)
            // target_chain is top-down but cur_chain is in
            // reverse
            if (target_chain[i] != cur_chain[cur_chain.size() - i - 1]) {
                matched = false;
                break;
            }
        if (matched)
            return true;
    }
    return false;
}

bool Reciprocity::is_reciprocity(const OpVec &oplist) {
    if (oplist.size() > (size_t)MAX_RECIPROCITY_DETECT_DEPTH)
        return false;
    std::vector<uint64_t> cur_chain;
    if (oplist.empty())
        return false;
    auto cur_op_it = oplist.back();
    for (size_t i = 0; i < oplist.size(); ++i) {
        if (!cur_op_it || cur_op_it->getType() != Operator::OpType::Transpose)
            break;
        cur_chain.push_back(cur_op_it->getHash());
        cur_op_it = cur_op_it->getInputs()[0]->getOutputOf();
    }
    if (cur_chain.empty())
        return false;
    for (auto &target_chain : reciprocal_op_chains) {
        if (target_chain.size() != cur_chain.size())
            continue;
        bool matched = true;
        for (int i = 0; i < (int)target_chain.size(); ++i)
            // target_chain is top-down but cur_chain is in
            // reverse
            if (target_chain[i] != cur_chain[cur_chain.size() - i - 1]) {
                matched = false;
                break;
            }
        if (matched)
            return true;
    }
    return false;
}