#include "code_gen/nnet/ReplaceKit.h"
#include "code_gen/nnet/Visitor/ReplaceVariable.h"
#include "code_gen/nnet/Visitor/SimplifyFormulaMutator.h"

namespace nnet {

RangeOp ReplaceKit::replaceRangeOpIterator(const RangeOp &rangeOp,
                                           const Replace &replace,
                                           const Expr &replacedSummand) {
    vector<VarRangePair> newVarRangePairs(replace.newVarRanges);
    if (replace.iteratorType == IterationType::Loop) {
        for (const auto &[var, range] : rangeOp->getLoopVarRanges()) {
            if (!replace.isReplaced(var))
                newVarRangePairs.emplace_back(var, range);
        }
        assert(newVarRangePairs.size() == rangeOp->getLoopVarRanges().size() -
                                              replace.oldIters.size() +
                                              replace.newIters.size());
        // Check the number of loop iterators
        return makeRangeOperator(newVarRangePairs, rangeOp->getSumVarRanges(),
                                 replacedSummand);
    } else if (replace.iteratorType == IterationType::Sum) {
        for (const auto &[var, range] : rangeOp->getSumVarRanges()) {
            if (!replace.isReplaced(var))
                newVarRangePairs.emplace_back(var, range);
        }
        assert(newVarRangePairs.size() == rangeOp->getSumVarRanges().size() -
                                              replace.oldIters.size() +
                                              replace.newIters.size());
        return makeRangeOperator(rangeOp->getLoopVarRanges(), newVarRangePairs,
                                 replacedSummand, rangeOp->getPaddings());
    }
    assert(false);
    return nullptr;
}

Subscript ReplaceKit::buildSubscirptForLoopVarReplace(const RangeOp &inner,
                                                      const Replace &replace) {
    VecExpr subs(replace.phis);
    for (size_t i = 0; i < replace.newVarRanges.size(); ++i) {
        assert(replace.newIters[i]->equal(inner->getLoopVar(i)));
    }
    for (size_t i = replace.newVarRanges.size();
         i < inner->getLoopVarRanges().size(); ++i) {
        subs.emplace_back(inner->getLoopVar(i));
    }
    // The support of var reorder and replace at the same time
    // VecExpr subs;
    // for (size_t i = 0; i < inner->getLoopVarRanges().size(); ++i) {
    //     if (auto it = std::find(replace.newIters.begin(),
    //                             replace.newIters.end(),
    //                             inner->getLoopVar(i));
    //         it != replace.newIters.end()) {
    //         subs.emplace_back(replace.phis[it - replace.newIters.begin()]);
    //     } else
    //         subs.emplace_back(inner->getLoopVar(i));
    // }
    return makeSubscript(inner, subs);
}

RangeOp
ReplaceKit::buildDLTOuterRangeOp(const RangeOp &original,
                                 const Subscript &subscriptedNewRangeOp) {
    auto outer = make_ref<RangeOpNode>(*original);
    outer->setSummand(subscriptedNewRangeOp);
    outer->setSumIterator({});
    return outer;
}

Expr ReplaceKit::replaceMultipleExprs(const Expr &cur,
                                      const vector<Var> &patterns,
                                      const VecExpr &replacements,
                                      bool simplify) {
    auto ret = cur;
    for (size_t i = 0; i < patterns.size(); ++i) {
        ret = replaceExpr(ret, patterns[i], replacements[i]);
    }
    if (simplify) {
        SimplifyFormulaMutator simplifyFormulaMutator;
        ret = simplifyFormulaMutator.simplify(ret);
    }
    return ret;
}

Expr ReplaceKit::replaceExpr(const Expr &cur, const Expr &pattern,
                             const Expr &replacement) {
    auto replace = ReplaceVariable(pattern, replacement);
    auto ret = replace(cur);
    return ret;
}

} // namespace nnet