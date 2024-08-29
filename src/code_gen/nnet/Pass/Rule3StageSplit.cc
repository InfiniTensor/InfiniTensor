#include "code_gen/nnet/Pass/Rule3StageSplit.h"
#include "code_gen/nnet/permutation.h"

namespace nnet {

void Rule3StageSplit::transform(Formula &origin, int depth, Expr &rCur) {
    auto cur = as<RangeOpNode>(rCur);
    vector<vector<Iterator>> splitSchemes = getSplitSummationIters(cur);

    for (const auto &varSplit : splitSchemes) {

        bool isSplittable = false;
        for (const auto &splitted : varSplit)
            if (cur->hasSumVar(splitted))
                isSplittable = true;
        assert(isSplittable);

        const vector<VarRangePair> loopVars = cur->getLoopVarRanges(),
                                   sumVars = cur->getSumVarRanges();
        // move iterators from Sigma to Loop
        vector<VarRangePair> innerLoopVars, innerSumVars, outerSumVars;
        VecExpr indexForInner;
        for (const auto &kv : sumVars) {
            bool isSplitted = false;
            for (const auto &iter : varSplit)
                if (iter == kv.first->getName())
                    isSplitted = true;
            if (isSplitted) {
                innerLoopVars.emplace_back(kv);
                outerSumVars.emplace_back(kv);
            } else
                innerSumVars.emplace_back(kv);
        }
        innerLoopVars.insert(innerLoopVars.end(), loopVars.begin(),
                             loopVars.end());
        for (const auto &[var, _] : innerLoopVars)
            indexForInner.emplace_back(var);

        // if no sum iterator, the stage is redundant
        assert(!innerSumVars.empty());
        auto inner =
            makeRangeOperator(innerLoopVars, innerSumVars, cur->getSummand());
        auto subscriptedInner = make_ref<SubscriptNode>(inner, indexForInner);
        auto outer = makeRangeOperator(cur->getLoopVarRanges(), outerSumVars,
                                       subscriptedInner);
        outer->setPaddings(cur->getPaddings());

        // next searching step
        string msg = "====== END rule3 Stage split: Move sum iterators {";
        for (const auto &iter : varSplit)
            msg += iter->getName() + " ";
        msg += "} to loop iterators\n";
        dbg(msg);
        msg = "Separate sum iters: " + serializeVec(varSplit);
        nextStep(origin, depth, rCur, outer, msg);
    }
}

vector<vector<Iterator>>
Rule3StageSplit::getSplitSummationIters(RangeOp rangeOp) {
    // set<string> varSplit = {"r", "s", "i3", "i13"};
    vector<vector<Iterator>> ret;
    // Rule-based Hint
    // vector<vector<Iterator>> heuristics = {{"r", "s"}, {"i3", "i13"}};
    // for (const auto &iterSet : heuristics) {
    //     bool notExist = false;
    //     for (const auto &iter : iterSet)
    //         if (!rangeOp->hasSumVar(iter))
    //             notExist = true;
    //     if (!notExist)
    //         ret.emplace_back(iterSet);
    // }
    // if (!rulesOverall.empty())
    //     return ret;
    vector<Iterator> sumIters;
    for (const auto &[iter, range] : rangeOp->getSumVarRanges())
        sumIters.emplace_back(iter);
    if (sumIters.size() <= 1)
        return ret;
    SubsetGenerator gen(sumIters);
    do {
        ret.emplace_back(gen.get());
    } while (gen.next());
    return ret;
}

} // namespace nnet