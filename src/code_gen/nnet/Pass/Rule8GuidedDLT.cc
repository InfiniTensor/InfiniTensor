#include "code_gen/nnet/Pass/Rule8GuidedDLT.h"
#include "code_gen/nnet/Visitor/ReplaceNodeMutator.h"

namespace nnet {

static int bitCount(unsigned int n) {
    int count = 0;
    while (n != 0) {
        n = n & (n - 1);
        count++;
    }
    return count;
}

static int bitPosition(unsigned int n) {
    assert(bitCount(n) == 1);
    int ret = 0;
    for (n >>= 1; n; n >>= 1)
        ++ret;
    return ret;
}

void Rule8GuidedDLT::transform(Formula &origin, int depth, Expr &rCur) {
    guidedDLT(origin, depth, rCur);
}

VecExpr Rule8GuidedDLT::guidedDLT(Formula &origin, int depth, Expr &rCur,
                                  bool debug) {
    string detailedMsg;
    VecExpr ret;
    auto cur = as<RangeOpNode>(rCur);
    // check cur satisfies T1[A]*T2[B]

    if (!statisfyGuidedDLT(cur))
        return ret;

    IteratorTable exprIT;
    if (!exprIT.analyzeExpr(cur))
        return ret;
    exprIT.buildTableWithDefaultMap();

    bool setTargetOpHere = false;
    for (int i = 0; i < MatchableRoutineTypeCnt; ++i) {
        // if not correctly unset this variable
        assert(setTargetOpHere == false);
        // If the guide direction is set
        if (derivator.getTargetOp() != RoutineType::NoneType &&
            idToRoutineType(i) != derivator.getTargetOp())
            continue;
        // Warning: no continue befor unset the targetOp
        if (derivator.getTargetOp() == RoutineType::NoneType) {
            setTargetOpHere = true;
            derivator.setTargetOp(idToRoutineType(i));
        }
        const Pattern &pattern = getPattern(derivator.getTargetOp());
        auto mismatches = exprIT.matchPatternIT(pattern);
        // Pruning less possible results

        // std::cout << "mismatches= " << mismatches.size()
        //           << "; setTargetOpHere: " << setTargetOpHere << "; ";
        // std::cout << "TargetOp = " <<
        // static_cast<int>(derivator.getTargetOp())
        //           << "; mismatches : ";
        // for (const auto i : mismatches)
        //     std::cout << static_cast<int>(i.type) << " ";
        // std::cout << endl;
        if (mismatches.size() == 0) {
            derivator.setSearchState(2);
            nextStep(origin, depth, rCur, rCur);
            derivator.setSearchState(1);
        }
        if (mismatches.size() > 0 && mismatches.size() <= 2) {
            for (const auto &mismatch : mismatches) {
                Expr newCur;
                if (mismatch.type == MismatchType::MoreVar) {
                    newCur = guidedDLTMoreVar2(cur, mismatch, exprIT, pattern);
                    detailedMsg += "guidedDLTMoreVar2 ";
                } else if (mismatch.type == MismatchType::DLMismatch ||
                           mismatch.type == MismatchType::OutputDLMismatch) {
                    if (mismatches.size() > 1) {
                        nnet_unimplemented_continue();
                        break;
                    }
                    newCur =
                        guidedDLTDLMismatch(cur, mismatch, exprIT, pattern);
                    detailedMsg += "guidedDLTDLMismatch ";
                }
                // std::cout << "newCur= "
                //           << ((newCur == nullptr) ? "Nullptr"
                //                                   : newCur->toReadable())
                //           << endl;
                if (!newCur)
                    continue;
                if (debug)
                    ret.emplace_back(newCur);
                // next searching step
                auto msg = "====== END guided rule: Guided DLT toward " +
                           getPatternName(derivator.getTargetOp()) + "\n";
                dbg(msg);
                detailedMsg = "Toward " +
                              getPatternName(derivator.getTargetOp()) + ". " +
                              detailedMsg;
                nextStep(origin, depth, rCur, newCur, detailedMsg);
            }
        }
        // Unset targetOp
        if (setTargetOpHere) {
            derivator.setTargetOp(RoutineType::NoneType);
            setTargetOpHere = false;
        }
    }
    return ret;
}

Expr Rule8GuidedDLT::guidedDLTDLMismatch(
    const RangeOp &cur, const Mismatch &mismatch,
    [[maybe_unused]] const IteratorTable &exprIT, const Pattern &pattern) {
    assert(mismatch.type == MismatchType::DLMismatch ||
           mismatch.type == MismatchType::OutputDLMismatch);
    // Currently only deal with ouput DLT
    if (mismatch.bitmap != pattern.getNumInputs()) {
        nnet_unimplemented_continue();
        return nullptr;
    }
    vector<VarRangePair> newVarRanges;
    for (const auto &[var, _] : pattern.getRangeOp()->getLoopVarRanges()) {
        const auto &iterInExpr = mismatch.mappingIter_r.at(var);
        newVarRanges.emplace_back(cur->getVarRange(iterInExpr));
    }
    auto inner = make_ref<RangeOpNode>(*cur);
    inner->setLoopIterator(newVarRanges);
    auto subscriptedInner =
        ReplaceKit::buildSubscirptForLoopVarReplace(inner, {});
    auto outer = ReplaceKit::buildDLTOuterRangeOp(cur, subscriptedInner);
    return outer;
}

bool Rule8GuidedDLT::statisfyGuidedDLT(RangeOp cur) const {
    auto mul = as<BinaryOpNode>(cur->getSummand());
    if (!mul)
        return false;
    if (mul->getOpType() != OpType::Mul)
        return false;
    return as<SubscriptNode>(mul->getLhs()) && as<SubscriptNode>(mul->getRhs());
}

Expr Rule8GuidedDLT::guidedDLTMoreVar2(const RangeOp &cur,
                                       const Mismatch &mismatch,
                                       const IteratorTable &exprIT,
                                       const Pattern &pattern) {
    int bitmap = mismatch.bitmap;
    const auto &mergedItersDefaultOrder = exprIT.getPosTable(bitmap);

    // Assure vars only appear in one input tensor
    int bitmapOfInputs = bitmap & ((1 << exprIT.getNumInputs()) - 1);
    if (bitCount(bitmapOfInputs) > 1)
        return nullptr;
    if (pattern.getPosTable(bitmap).size() != 1) {
        nnet_unimplemented_continue();
        return nullptr;
    }
    if (mergedItersDefaultOrder.size() < 1)
        return nullptr;
    int tensorID = bitPosition(bitmapOfInputs);
    if (!checkElementsHaveOnlyOneAccessIteratorSet(exprIT, tensorID))
        return nullptr;
    vector<Var> oldVars; // i_1, ...
    vector<Var> newVars; // j_1, ...
    VecExpr psis;        // i_1=\psi_1(j_1, ...)
    VecExpr phis;        // j_1=\phi_1(i_1, ...), not necessary for Sum iter
    vector<VarRangePair> newVarRanges;

    auto originalTensor = exprIT.getTensor(tensorID);
    auto originalSub = exprIT.getSubscript(tensorID);
    vector<bool> mergedDims(originalTensor->getDims());

    // Heuristic: merge iters according to their appearance positions
    std::multimap<int, Var> sortedMergedIters;
    for (const auto &iter : mergedItersDefaultOrder) {
        vector<int> dims = exprIT.getIterDimInTensor(tensorID, iter);
        assert(dims.size() == 1);
        sortedMergedIters.emplace(dims[0], iter);
    }
    vector<Var> mergedIters; // decides the order of fused dims
    for (const auto &[_, v] : sortedMergedIters)
        mergedIters.emplace_back(v);

    // Add the merged iterators
    const auto newVar = getNewVar();
    newVars.emplace_back(newVar);
    int newRange = 1;
    for (const auto &iter : mergedIters) {
        oldVars.emplace_back(iter);
        auto range = cur->getRange(iter);
        newRange *= (range.second - range.first);
        // if (range.first == 0)
        //     nnet_unimplemented_halt();
    }
    newVarRanges.emplace_back(newVar, Range{0, newRange});
    // Add psis for each old iterator
    int remainingRange = newRange;
    Expr phi = nullptr;
    for (const auto &iter : mergedIters) {
        auto oldVar = iter;
        auto range = cur->getRange(iter);
        int len = (range.second - range.first);
        remainingRange /= len;
        Expr psi = newVar;
        if (remainingRange > 1)
            psi = psi / remainingRange;
        if (newRange > remainingRange * len)
            psi = psi % len;
        int start = cur->getRange(iter).first;
        if (start != 0)
            psi = psi + start;
        psis.emplace_back(psi);
        phi = phi + remainingRange * (oldVar - start);
    }
    Replace replace{.iteratorType = IterationType::Loop,
                    .oldIters = oldVars,
                    .newIters = newVars,
                    .phis = VecExpr{phi},
                    .psis = psis,
                    .newVarRanges = newVarRanges};
    // HACK: decide the rebuild data shape order
    // TODO: get a partial iter mapping and permutate them?
    vector<Var> tensorDimAxes{newVars};
    vector<int> newShape;
    for (const auto &[var, range] : newVarRanges)
        newShape.emplace_back(range.second - range.first);
    for (int row = 0; row < exprIT.getNumRows(); ++row) {
        // Deal with other dimensions of the current tensor
        if (row == bitmap || ((row & (1 << tensorID)) == 0))
            continue;
        using StrideIter = tuple<int, int, Iterator>;
        vector<StrideIter> strideIters;

        for (size_t i = 0; i < exprIT.getPosTable(row).size(); ++i) {
            const auto &iter = exprIT.getPosTable(row)[i];
            const Range range = cur->getRange(iter);
            const int len = range.second - range.first;

            // HACK Sort according to original stride. (keep original order)
            strideIters.emplace_back(-exprIT.getStridesInTensor(iter, tensorID),
                                     len, iter);

            // // HACK for conv
            // if (iter == "n")
            //     strideIters.emplace_back(2, len, iter);
            // else if (iter == "c")
            //     strideIters.emplace_back(1, len, iter);
            // else
            //     strideIters.emplace_back(0, len, iter);
        }
        // HACK: Assure the order of iterators
        std::sort(strideIters.begin(), strideIters.end(),
                  ref_value_less<StrideIter>);
        for (const auto &[_, len, oldIter] : strideIters) {
            const auto &oldVar = oldIter;
            tensorDimAxes.emplace_back(oldVar);
            newShape.emplace_back(len);
        }
    }

    // build DLT source
    const auto sourceExpr =
        buildGuidedDLTSource(originalSub, replace, tensorDimAxes, newShape);
    const auto sourceRoutine = make_ref<ElementWiseNode>(
        sourceExpr, vector<Tensor>{originalTensor}, newShape);
    // build stage connections
    const auto newTensor =
        makeTensor(newTensorName(), newShape, {}, sourceRoutine);
    const auto &newSub = makeSubscript(
        newTensor, VecExpr(tensorDimAxes.begin(), tensorDimAxes.end()));
    // TODO [1124]: get variable mapping and reorder L according to it
    // dbg(cur, originalSub, newSub, newVarRanges, replace.toReadable(),
    //     tensorDimAxes, newShape);

    // Replace the entire subscript(A[xxxxx,xxx]) in the summand
    Expr newSummand = ReplaceNodeMutator().replace(cur->getSummand(),
                                                   originalSub.get(), newSub);
    auto inner = ReplaceKit::replaceRangeOpIterator(cur, replace, newSummand);
    auto subscriptedInner =
        ReplaceKit::buildSubscirptForLoopVarReplace(inner, replace);
    auto outer = ReplaceKit::buildDLTOuterRangeOp(cur, subscriptedInner);
    return outer;
}

bool Rule8GuidedDLT::checkElementsHaveOnlyOneAccessIteratorSet(
    const IteratorTable &exprIT, int tensorID) {
    const auto &strideInDim = exprIT.getStrideInDim();
    for (const auto &strideForOneDim : strideInDim[tensorID]) {
        vector<pair<int, int>> strideLengthPairs;
        for (const auto &[iter, s] : strideForOneDim) {
            const auto &range = exprIT.getRangeOp()->getRange(iter);
            strideLengthPairs.emplace_back(s, range.second - range.first);
        }
        std::sort(strideLengthPairs.begin(), strideLengthPairs.end());
        for (size_t i = 0; i < strideLengthPairs.size() - 1; ++i) {
            const auto &[stride, length] = strideLengthPairs[i];
            if (stride * length > strideLengthPairs[i + 1].first)
                return false;
        }
    }
    return true;
}

Expr Rule8GuidedDLT::buildGuidedDLTSource(const Subscript &originalSub,
                                          Replace replace,
                                          vector<Var> tensorDimAxes,
                                          vector<int> newShape) {
    Expr newSub = ReplaceKit::replaceMultipleExprs(
        originalSub, replace.oldIters, replace.psis, true);
    vector<VarRangePair> loopVarRangePairs;
    for (size_t i = 0; i < tensorDimAxes.size(); ++i)
        loopVarRangePairs.emplace_back(tensorDimAxes[i], pair(0, newShape[i]));
    return makeRangeOperator(loopVarRangePairs, {}, newSub);
}

} // namespace nnet