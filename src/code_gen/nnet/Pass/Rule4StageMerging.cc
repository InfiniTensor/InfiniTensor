#include "code_gen/nnet/Pass/Rule4StageMerging.h"
#include "code_gen/nnet/Visitor/ReplaceNodeMutator.h"
#include "code_gen/nnet/Visitor/ReplaceVariable.h"

namespace nnet {

void Rule4StageMerging::transform(Formula &origin, int depth, Expr &rCur) {
    success = rule4StageMerging(origin, depth, rCur, mergeStageWithCalc);
}

bool Rule4StageMerging::isSuccessful() { return success; }

void Rule4StageMerging::setMergeStageWithCalc(bool value) {
    mergeStageWithCalc = value;
}

bool Rule4StageMerging::rule4StageMerging(Formula &origin, int depth,
                                          Expr &rCur, bool mergeStageWithCalc) {
    auto rangeOp0 = as<RangeOpNode>(rCur);
    const Subscript &sub0 = as<SubscriptNode>(rangeOp0->getSummand());
    if (!sub0)
        return false;
    const auto &rangeOp1 = as<RangeOpNode>(sub0->getObject());
    if (!rangeOp1)
        return false;
    const auto &sub1 = as<SubscriptNode>(rangeOp1->getSummand());
    if (!sub1)
        return false;
    // merge stage with calculation only when mergeStageWithCalc=true
    if (!mergeStageWithCalc && !rangeOp1->getSumVarRanges().empty())
        return false;
    // Only propogate paddings in perfect nested dimension
    if (rangeOp1->hasPaddings()) {
        auto oldTensor = as<TensorNode>(sub1->getObject());
        if (!oldTensor) {
            nnet_unimplemented_continue();
            return 0;
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
                return false;
            }
        }
        merged = as<RangeOpNode>(
            ReplaceNodeMutator().replace(merged, oldTensor.get(), newTensor));
        assert(merged != nullptr);
    }
    // Merge inner stage sums
    if (!rangeOp1->getSumVarRanges().empty())
        merged->setSumIterator(rangeOp1->getSumVarRanges());

    // next searching step
    string msg = "====== END rule4StageMerging\n";
    dbg(msg);
    // if mergeStageWithCalc, depth counts for invocation in rule-based search
    nextStep(origin, (mergeStageWithCalc) ? depth : depth - 1, rCur, merged);
    return true;
}

} // namespace nnet