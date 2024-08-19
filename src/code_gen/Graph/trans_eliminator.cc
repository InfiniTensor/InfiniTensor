#include "code_gen/trans_eliminator.h"

namespace tpm {
TransEliminator::TransEliminator() {
    // TODO: maintain trans list
    // dilated
    all_ops.emplace_back(new tpm::TransposeOp(2, {0, 1, {-1, 2}, 3}, 2));
    all_ops.emplace_back(new tpm::TransposeOp(3, {0, 1, 2, {-1, 3}}, 2));
    all_ops.emplace_back(new tpm::TransposeOp(2, {0, 1, {-1, 2}, 3}, -2));
    all_ops.emplace_back(new tpm::TransposeOp(3, {0, 1, 2, {-1, 3}}, -2));
    // n->h
    all_ops.emplace_back(new tpm::TransposeOp(0, {0, 1, {-1, 2}, 3}, 2));
    all_ops.emplace_back(new tpm::TransposeOp(2, {{0, 2}, 1, -1, 3}, -2));
    // n->w
    all_ops.emplace_back(new tpm::TransposeOp(0, {0, 1, 2, {-1, 3}}, 2));
    all_ops.emplace_back(new tpm::TransposeOp(3, {{0, 3}, 1, 2, -1}, -2));
    // c->h
    all_ops.emplace_back(new tpm::TransposeOp(1, {0, 1, {2, -1}, 3}, 2));
    // c->w
    all_ops.emplace_back(new tpm::TransposeOp(1, {0, 1, 2, {3, -1}}, 2));
    reciprocity = std::make_shared<Reciprocity>(all_ops);
}

std::shared_ptr<SubGraph>
TransEliminator::eliminate(std::shared_ptr<SubGraph> &graph) {
    if (!checkValid(graph))
        return graph;
    std::shared_ptr<SubGraph> ret = graph;
    std::shared_ptr<SubGraph> cur = graph;
    while (doEliminate(cur, ret) != 0) {
        cur = ret;
    }
    return ret;
}

bool TransEliminator::checkValid(std::shared_ptr<SubGraph> &graph) {
    if (graph == nullptr)
        return false;
    if (graph->getInputs().size() != 1)
        return false;
    auto input = graph->getInputs()[0];
    if (input->getInputOf().size() != 1)
        return false;
    if (graph->getOperators().empty())
        return false;
    auto curOp = input->getInputOf()[0];
    while (true) {
        assert(curOp != nullptr);
        if (curOp->getType() != Operator::Transpose)
            return false;
        if (curOp->getSuccessors().empty())
            break;
        if (curOp->getSuccessors().size() > 1)
            return false;
        curOp = curOp->getSuccessors()[0];
    }
    return true;
}

int TransEliminator::doEliminate(std::shared_ptr<SubGraph> &graph,
                                 std::shared_ptr<SubGraph> &eliminated) {
    if (graph == nullptr || graph->getOperators().size() == 0)
        return 0;
    OpVec orderedOpList;
    auto curOp = graph->getInputs()[0]->getInputOf()[0];
    orderedOpList.emplace_back(curOp);
    while (!curOp->getSuccessors().empty()) {
        curOp = curOp->getSuccessors()[0];
        orderedOpList.emplace_back(curOp);
    }
    auto sz = orderedOpList.size();
    for (size_t segEnd = 1; segEnd <= sz; ++segEnd) {
        for (size_t segBeg =
                 std::max(0ul, segEnd - reciprocity->maxDetectDepth());
             segBeg < segEnd; ++segBeg) {
            auto segOps = OpVec(orderedOpList.begin() + segBeg,
                                orderedOpList.begin() + segEnd);
            if (reciprocity->is_reciprocity(segOps)) {
                if (segEnd - segBeg == sz) {
                    eliminated = nullptr;
                    return sz;
                } else if (segEnd < sz) {
                    orderedOpList[segEnd]->getInputs() =
                        orderedOpList[segBeg]->getInputs();
                } else if (segBeg > 0) {
                    orderedOpList[segBeg - 1]->getOutputs() =
                        orderedOpList[segEnd - 1]->getOutputs();
                }
                OpVec restOp;
                for (size_t k = 0; k < sz; ++k) {
                    if (k < segBeg || k >= segEnd)
                        restOp.emplace_back(orderedOpList[k]);
                }
                eliminated = std::make_shared<SubGraph>(restOp);
                return segEnd - segBeg;
            }
        }
    }
    return 0;
}
} // end of namespace tpm
