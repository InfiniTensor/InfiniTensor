#include "code_gen/nnet/Visitor/MergeMemboundMutator.h"
#include "code_gen/nnet/Visitor/CheckOOBVisitor.h"
#include "code_gen/nnet/Visitor/ReplaceNodeMutator.h"
#include "code_gen/nnet/Visitor/ReplaceVariable.h"

namespace nnet {

Expr MergeMemboundMutator::merge(bool allowEmptyMembound) {
    // FIXME: fix empty expression in membound
    assert(kernels.size() > 1);
    if (checkEmpty()) {
        if (allowEmptyMembound)
            return nullptr;
        else
            nnet_assert(false, "Empty membound expression");
    }
    // Nesting stages
    auto expr = dispatch(kernels.back());
    // Fusing stages
    bool merged = false;
    do {
        merged = false;
        RangeOp curRangeOp;
        for (Expr *curExpr = &expr;
             curExpr && (curRangeOp = as<RangeOpNode>(*curExpr));) {
            auto curRangeOp = as<RangeOpNode>(*curExpr);
            assert(CheckOOBVisitor().checkRangeOp(curRangeOp) == false);
            auto summand = curRangeOp->getSummand();
            if (auto subscriptOp = as<SubscriptNode>(summand)) {
                if (auto mergedExpr = rule4StageMerging(*curExpr, true)) {
                    // dbg(*curExpr, mergedExpr);
                    *curExpr = mergedExpr;
                    merged = true;
                    break;
                }
                curExpr = subscriptOp->getObjectPtr();
                nnet_assert(*curExpr != nullptr, __LINE__);
            } else if (auto funcOp = as<FuncNode>(summand)) {
                // Relu({...}[i,j])
                curExpr = funcOp->getObject()->getObjectPtr();
            } else
                nnet_unimplemented_halt();
        }
    } while (merged);
    return expr;
}

bool MergeMemboundMutator::checkEmpty() {
    for (const auto &k : kernels) {
        if (k == nullptr)
            return true;
    }
    return false;
}

Expr MergeMemboundMutator::visit_(const Tensor &c) {
    if (curDepth > 0)
        return dispatch(kernels[--curDepth]);
    else {
        // Reach the last tensor, return it to reconstruct the total tree
        return c;
    }
}

Expr MergeMemboundMutator::rule4StageMerging(Expr &rCur,
                                             bool mergeStageWithCalc) {
    auto rangeOp0 = as<RangeOpNode>(rCur);
    const Subscript &sub0 = as<SubscriptNode>(rangeOp0->getSummand());
    if (!sub0)
        return nullptr;
    const auto &rangeOp1 = as<RangeOpNode>(sub0->getObject());
    if (!rangeOp1)
        return nullptr;
    const auto &sub1 = as<SubscriptNode>(rangeOp1->getSummand());
    if (!sub1)
        return nullptr;
    // merge stage with calculation only when mergeStageWithCalc=true
    if (!mergeStageWithCalc && !rangeOp1->getSumVarRanges().empty())
        return nullptr;
    // Only propogate paddings in perfect nested dimension
    if (rangeOp1->hasPaddings()) {
        auto oldTensor = as<TensorNode>(sub1->getObject());
        if (!oldTensor) {
            nnet_unimplemented_continue();
            return nullptr;
        }
    }
    // repalce variables: iters of rangeOp1 repalced by indexes of sub0
    map<string, pair<Expr, Expr>> varMapping;
    assert(sub0->getDims() == rangeOp1->getLoopVarRanges().size());
    for (size_t i = 0; i < sub0->getDims(); ++i) {
        varMapping[rangeOp1->getLoopVar(i)->getName()] =
            pair(rangeOp1->getLoopVar(i), sub0->getIndex(i));
    }
    ReplaceVariable replaceVariable{varMapping};
    auto merged = make_ref<RangeOpNode>(*rangeOp0);
    merged->setSummand(replaceVariable(sub1));
    // a naive approach to propogate paddings
    if (rangeOp1->hasPaddings()) {
        auto oldTensor = as<TensorNode>(sub1->getObject());
        auto newTensor = make_ref<TensorNode>(*oldTensor);
        for (int i = 0; i < rangeOp1->getNumOutputDims(); ++i) {
            if (rangeOp1->getPaddings(i) == 0)
                continue;
            auto loopVar = rangeOp1->getLoopVar(i);
            // FIXME: in fact this var should not appear in other index as well,
            // which may result in OOB
            bool findSingleVarAsIndex = false;
            for (size_t subIndexID = 0; subIndexID < sub1->getDims();
                 ++subIndexID) {
                auto index = sub1->getIndex(subIndexID);
                if (auto indexVar = as<VarNode>(index);
                    indexVar && (indexVar->equal(loopVar))) {
                    newTensor->setPadding(subIndexID,
                                          newTensor->getPadding(subIndexID) +
                                              rangeOp1->getPaddings(i));
                    findSingleVarAsIndex = true;
                }
            }
            if (!findSingleVarAsIndex) {
                nnet_unimplemented_continue();
                return nullptr;
            }
        }
        merged = as<RangeOpNode>(
            ReplaceNodeMutator().replace(merged, oldTensor.get(), newTensor));
        assert(merged != nullptr);
    }
    // Merge inner stage sums
    if (!rangeOp1->getSumVarRanges().empty())
        merged->setSumIterator(rangeOp1->getSumVarRanges());
    return merged;
}

} // namespace nnet