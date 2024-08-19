#include "code_gen/nnet/Pass/Rule2VariableMerging.h"
#include "code_gen/nnet/Visitor/CheckOOBVisitor.h"

namespace nnet {

void Rule2VariableMerging::transform(Formula &origin, int depth, Expr &rCur) {
    // Extract r and s
    auto cur = as<RangeOpNode>(rCur);
    vector<Replace> replaces = getMergableReplaces(cur, depth);
    // dbg("Start rule2VariableMerging", depth, mergableExprs, *cur);
    for (const auto &replace : replaces) {
        if (replace.iteratorType != IterationType::Loop) {
            nnet_unimplemented_continue();
            continue;
        }
        // replace vars in summand
        auto replacedSummand = ReplaceKit::replaceMultipleExprs(
            cur->getSummand(), replace.oldIters, replace.psis, true);
        // replace var in rangeOp
        auto inner =
            ReplaceKit::replaceRangeOpIterator(cur, replace, replacedSummand);
        // If OOB happens, this transformation is skipped
        if (CheckOOBVisitor().checkRangeOp(inner))
            continue;
        // build the outerRange{innerRange}[indexForInner] to do DLT
        auto subscriptedInner =
            ReplaceKit::buildSubscirptForLoopVarReplace(inner, replace);
        auto outer = ReplaceKit::buildDLTOuterRangeOp(cur, subscriptedInner);

        // next searching step
        string msg = "====== END rule2VariableMerging: Replace " +
                     replace.toReadable() + "\n";
        dbg(msg);
        msg = replace.toReadable();
        nextStep(origin, depth, rCur, outer, msg);
    }
}

vector<Replace> Rule2VariableMerging::getMergableReplaces(RangeOp rangeOp,
                                                          int depth) {
    vector<Replace> ret;
    IteratorTable exprIT;
    if (!exprIT.analyzeExpr(rangeOp)) {
        nnet_unimplemented_continue();
        return ret;
    }
    exprIT.buildTableWithDefaultMap();
    const auto &strideInAllDim = exprIT.getStrideInDim();

    set<pair<Iterator, Iterator>, RefValueLess<pair<Iterator, Iterator>>>
        checkedIterPairs{};
    // strideInAllDim: [tensorID][dimOfTensor][Iterator]=stride
    for (size_t tensorID = 0; tensorID < strideInAllDim.size(); ++tensorID) {
        const auto &strideInDimsOfATensor = strideInAllDim[tensorID];
        for (const PtrMap<Iterator, int> &strideInADim :
             strideInDimsOfATensor) {
            for (const auto &it1 : strideInADim) {
                for (const auto &it2 : strideInADim) {
                    // Backdoor for rule-based search
                    if (substituteRules.count(depth)) {
                        if (substituteRules[depth].at(0)->neq(it1.first))
                            continue;
                        if (substituteRules[depth].at(1)->neq(it2.first))
                            continue;
                    }
                    if (!(it1.first->equal(it2.first) &&
                          it1.second == it2.second) &&
                        rangeOp->hasLoopVar(it1.first) &&
                        rangeOp->hasLoopVar(it2.first)) {
                        // 2 iters -> 2 iters
                        if (auto opt = getReplaceMappingTwoLoopIters(rangeOp,
                                                                     it1, it2))
                            ret.emplace_back(*opt);

                        // 2 iters -> 1 iter
                        const auto iterPair = pair(it1.first, it2.first);
                        if (!checkedIterPairs.count(iterPair)) {
                            checkedIterPairs.insert(iterPair);
                            if (auto opt = getReplaceMergingTwoLoopIters(
                                    rangeOp, it1, it2, exprIT, tensorID))
                                ret.emplace_back(*opt);
                        }
                    }
                }
            }
        }
    }
    return ret;
}

optional<Replace> Rule2VariableMerging::getReplaceMergingTwoLoopIters(
    const RangeOp &rangeOp, pair<Iterator, int> pairA,
    pair<Iterator, int> pairB, const IteratorTable &exprIT, int tensorID) {
    // 1*A + sb*B -> C
    // A=C%sb, B=C/sb
    // ax+by->z, a=1 or -1
    // For a>0 and b>0 : x=z%b, y=z/b
    auto x = pairA.first, y = pairB.first;
    int a = pairA.second, b = pairB.second;
    if (abs(a) != 1 || abs(a) * abs(b) <= 0)
        return {};
    if (a < 0 && b > 0) { // The only unhandled case
        nnet_unimplemented_continue();
        return {};
    }
    // negative substitution happens only if can be totally merged. So if the
    // variable appears in another index, skip it.
    if (a < 0 || b < 0) {
        if (exprIT.getNumInputs() > 1) {
            if (exprIT.getStridesInTensor(x, 1 - tensorID) != 0)
                return {};
            if (exprIT.getStridesInTensor(y, 1 - tensorID) != 0)
                return {};
        }
    }
    Range rangeX = rangeOp->getVarRange(x).second,
          rangeY = rangeOp->getVarRange(y).second;
    if (rangeX.first != 0 || rangeY.first != 0)
        return {};
    int lenX = rangeX.second - rangeX.first;
    if (abs(b) != lenX)
        return {};
    auto z = getNewVar();

    Range rangeExpr{0, 1}; // 1 is the open interval compensation
    auto calcRangeExpr = [&rangeExpr](int stride, const Range &r) {
        if (stride > 0) {
            rangeExpr.first += stride * r.first;
            rangeExpr.second += stride * (r.second - 1);
        } else {
            rangeExpr.first += stride * (r.second - 1);
            rangeExpr.second += stride * r.first;
        }
    };
    calcRangeExpr(a, rangeX);
    calcRangeExpr(b, rangeY);

    // build the phi/psi for index transformation
    // phi: j_x=(i_x...),  psi: i_x=(j_x...)
    auto ret = optional<Replace>();
    ret.emplace();
    ret->iteratorType = IterationType::Loop;
    ret->newIters = {z};
    ret->oldIters = {x, y};
    ret->phis = {a * x + b * y - rangeExpr.first};
    // For b < 0, the psis are not an equavalent replace. Since it must be
    // simplified (z/b and z%b will be merged), the only important thing is
    // their strides should be mergable. To merge the strides, an extra minus
    // are introduced if their stride is negative.
    ret->psis = {a * (z % b) + a * rangeExpr.first, (b > 0 ? 1 : -1) * (z / b)};
    ret->newVarRanges = {{z, {0, rangeExpr.second - rangeExpr.first}}};
    return ret;
}

optional<Replace>
Rule2VariableMerging::getReplaceMappingTwoLoopIters(const RangeOp &rangeOp,
                                                    pair<Iterator, int> pairA,
                                                    pair<Iterator, int> pairB) {
    // the first iterator is replaced, the second remains
    auto i1 = pairA.first, i2 = pairB.first;
    int sa = pairA.second, sb = pairB.second;
    // TODO: can be relaxed to sb|sb
    if (sa != 1 || sb == 0)
        return {};
    if (sb < 0) {
        nnet_unimplemented_continue();
        return {};
    }
    Range rangeA = rangeOp->getVarRange(i1).second;
    Range rangeB = rangeOp->getVarRange(i2).second;
    auto j1 = getNewVar(), j2 = getNewVar();
    Range rangeJ1, rangeJ2 = rangeB;
    assert(pairA.second == 1);
    rangeJ1.first = rangeA.first + rangeB.first * sb;
    rangeJ1.second = rangeA.second + (rangeB.second - 1) * sb;
    // build the phi/psi for index transformation
    // phi: j_x=(i_x...),  psi: i_x=(j_x...)
    auto ret = optional<Replace>();
    ret.emplace();
    ret->iteratorType = IterationType::Loop;
    ret->newIters = {j1, j2};
    ret->oldIters = {i1, i2};
    ret->newVarRanges = {{j1, rangeJ1}, {j2, rangeJ2}};
    ret->phis = {sa * i1 + sb * i2, i2};
    ret->psis = {j1 - (sb / sa) * j2, j2};
    return ret;
}

} // namespace nnet