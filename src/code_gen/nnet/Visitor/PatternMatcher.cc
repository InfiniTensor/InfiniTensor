#include "code_gen/nnet/Visitor/PatternMatcher.h"
#include "code_gen/nnet/ReplaceKit.h"
#include "code_gen/nnet/Visitor/MatmulTransposeMutator.h"
#include "code_gen/nnet/Visitor/ReplaceVariable.h"

namespace nnet {

PatternMatcher::PatternMatcher(Derivator &derivator, const RangeOp &cur)
    : Functor(false), derivator(derivator), hasNonZeroRange(false),
      originalCur(cur) {
    for (const auto &[var, range] : cur->getLoopVarRanges())
        if (range.first != 0)
            hasNonZeroRange = true;
}

RangeOp PatternMatcher::getOffsetCur() {
    if (!hasNonZeroRange)
        return originalCur;
    vector<Var> itersFromNonZero;
    VecExpr psis;
    vector<VarRangePair> newLoopVarRanges;
    for (const auto &[var, range] : originalCur->getLoopVarRanges()) {
        if (range.first == 0) {
            newLoopVarRanges.emplace_back(var, range);
        } else {
            auto newVar = derivator.getNewVar();
            newLoopVarRanges.emplace_back(newVar,
                                          pair(0, range.second - range.first));
            itersFromNonZero.emplace_back(var);
            psis.emplace_back(newVar + range.first);
        }
    }
    auto newSummand = ReplaceKit::replaceMultipleExprs(
        originalCur->getSummand(), itersFromNonZero, psis);
    return makeRangeOperator(newLoopVarRanges, originalCur->getSumVarRanges(),
                             newSummand);
}

VecExpr PatternMatcher::matchKernel(const Pattern &pattern,
                                    const RangeOp &rangeOp,
                                    IteratorTable &exprIT) {
    VecExpr ret;
    if (pattern.getNumTensors() != (int)exprIT.getNumTensors())
        return ret;

    // Whether enable tensor permutation
    if (false) {
        const int nInputs = pattern.getNumInputs();
        vector<int> tensorMap; // [tensors Index] -> pattern tensor ID
        for (int i = 0; i < nInputs; ++i)
            tensorMap.emplace_back(i);
        do {
            exprIT.buildTable(tensorMap);
            auto matched = matchKernelWithTensorMap(pattern, rangeOp, exprIT);
            if (matched)
                ret.emplace_back(matched);
        } while (std::next_permutation(tensorMap.begin(), tensorMap.end()));
    } else {
        exprIT.buildTableWithDefaultMap();
        auto matched = matchKernelWithTensorMap(pattern, rangeOp, exprIT);
        if (matched)
            ret.emplace_back(matched);
    }
    // Generate 8 variants of gemm
    if (true) // Disabled for debug
        if (!ret.empty() && dynamic_cast<const MatmulPattern *>(&pattern)) {
            auto tensor = as<TensorNode>(ret[0]);
            auto transposeds =
                MatmulTransposeMutator(derivator).transpose(tensor);
            for (const auto &transposed : transposeds)
                ret.emplace_back(transposed);
        }
    return ret;
}

Expr PatternMatcher::matchKernelWithTensorMap(const Pattern &pattern,
                                              const RangeOp &rangeOp,
                                              IteratorTable &exprIT) {
    auto mismatches = exprIT.matchPatternIT(pattern);
    if (!mismatches.empty())
        return nullptr;

    const auto &[tensorMap_r, iterToRange_r] = exprIT.getReverseMap();
    // // TODO: check OOB error
    // for (int tensorID = 0; tensorID < pattern.getNumInputs(); ++tensorID) {
    //     if (!checkIndexOutOfBound(pattern.getIterInTensorDim(tensorID),
    //                      tensorMap_r[tensorID], iterToRange_r))
    //         return nullptr;
    // }

    // matched! build expr for ret;
    return pattern.buildExpr(rangeOp, tensorMap_r, iterToRange_r,
                             derivator.newTensorName(), exprIT);
}

VecExpr PatternMatcher::applyWrapper(const VecExpr &exprs) {
    if (!hasNonZeroRange)
        return exprs;
    VecExpr ret, indexes;
    for (const auto &[var, range] : originalCur->getLoopVarRanges()) {
        if (range.first == 0) {
            indexes.emplace_back(var);
        } else {
            hasNonZeroRange = true;
            indexes.emplace_back(var - range.first);
        }
    }
    for (auto &expr : exprs) {
        auto newSub = makeSubscript(expr, indexes);
        ret.emplace_back(makeRangeOperator(originalCur->getLoopVarRanges(), {},
                                           newSub, originalCur->getPaddings()));
    }
    return ret;
}

VecExpr PatternMatcher::matchWithPattern(const RangeOp &rangeOp,
                                         const Pattern &pattern) {
    IteratorTable exprIT;
    if (!exprIT.analyzeExpr(rangeOp))
        return {};
    return matchKernel(pattern, rangeOp, exprIT);
}

} // namespace nnet