#include "code_gen/nnet/Pass/Rule1VariableSplit.h"
#include "code_gen/nnet/Visitor/ReplaceVariable.h"

namespace nnet {

void Rule1VariableSplit::transform(Formula &origin, int depth, Expr &rCur) {
    auto cur = as<RangeOpNode>(rCur);
    vector<Replace> replaces = getSplitableVar(cur);
    // for (const auto &replace : replaces)
    //     dbg(replace.oldIters, replace.newIters, replace.psis,
    //         replace.newVarRanges);
    for (const auto &replace : replaces) {
        auto replacedSummand = replaceIters(cur->getSummand(), replace);
        if (!replacedSummand) {
            // TODO: if a real getMergableExprs is implemented, this case should
            // be an error. Since the expr should appear in the AST.
            dbg("Warning: No replacment happens.");
            continue;
        }
        auto inner =
            ReplaceKit::replaceRangeOpIterator(cur, replace, replacedSummand);
        // build the outerRange{innerRange}[indexForInner] to do DLT
        Expr nextCur = nullptr;
        if (replace.iteratorType == IterationType::Loop) {
            auto subscriptedInner =
                ReplaceKit::buildSubscirptForLoopVarReplace(inner, replace);
            nextCur = ReplaceKit::buildDLTOuterRangeOp(cur, subscriptedInner);
        } else
            nextCur = inner;

        string msg = "====== END rule1 VariableSplit: ";
        dbg(msg, replace.oldIters, replace.newIters, replace.phis,
            replace.psis);
        msg = replace.toReadable();
        nextStep(origin, depth, rCur, nextCur, msg);
    }
}

vector<Replace> Rule1VariableSplit::getSplitableVar(const RangeOp &rangeOp) {
    vector<Replace> ret;
    // Split strategy
    vector<int> SumFactors, LoopFactors;
    if (derivator.getPassMode() == Derivator::PassMode::Debug) {
        SumFactors = {3};
        LoopFactors = {4};
    } else if (derivator.getPassMode() == Derivator::PassMode::Full) {
        SumFactors = {2, 3};
        // LoopFactors = {3, 4};
        LoopFactors = {4};
    } else
        nnet_unimplemented_halt();

    // Split Sum variable
    for (const int k : SumFactors) {
        for (const auto &[var, range] : rangeOp->getSumVarRanges()) {
            int len = range.second - range.first;
            auto p1 = getNewVar(); // p1=i/k
            auto p2 = getNewVar(); // p2=i%k
            if (len > 10 || len <= k || len % k != 0)
                continue;

            Range range1, range2;
            if (range.first < 0) {
                nnet_unimplemented_halt();
                // FIXME: this must be ERROR
                range1.first = range.first / k;
                range1.second = range1.first + len / k;
                range2.first = -k / 2;
                range2.second = range2.first + k;
            } else if (range.first == 0) {
                range1.first = 0;
                range1.second = len / k;
                range2.first = 0;
                range2.second = k;
            } else {
                nnet_unimplemented_continue();
                continue;
            }
            Replace replace{.iteratorType = IterationType::Sum,
                            .oldIters = {var},
                            .newIters = {p1, p2},
                            .phis = {},
                            .psis = {make_ref<ConstantNode>(k) * p1 + p2},
                            .newVarRanges = {{p1, range1}, {p2, range2}}};
            ret.emplace_back(replace);
        }
    }
    for (const int k : LoopFactors) {
        // Split Loop variable
        for (const auto &[var, range] : rangeOp->getLoopVarRanges()) {
            const int len = range.second - range.first;
            // Debug HACK for dilated SG2BMM
            if (derivator.getPassMode() == Derivator::PassMode::Debug &&
                !(var->getName() == "m" && len % k == 0))
                continue;

            // Illeagel conditions
            if (range.second - range.first <= k ||
                (range.second - range.first) % k != 0)
                continue;
            // Unsupport conditions
            if (range.first != 0)
                continue;
            auto p1 = getNewVar(); // p1=i/k
            auto p2 = getNewVar(); // p2=i%k
            Range range1(0, len / k);
            Range range2(0, k);
            nnet_assert(range1.second > 0 && range2.second > 0,
                        "Empty loop dim");
            Replace replace{.iteratorType = IterationType::Loop,
                            .oldIters = {var},
                            .newIters = {p1, p2},
                            .phis = {var / 4, var % 4},
                            .psis = {make_ref<ConstantNode>(k) * p1 + p2},
                            .newVarRanges = {{p1, range1}, {p2, range2}}};
            ret.emplace_back(replace);
        }
    }
    return ret;
}

Expr Rule1VariableSplit::replaceIters(Expr cur, const Replace &replace) {
    // TODO [feature]: support multiple replacements in one mutator
    if (replace.oldIters.size() != 1) {
        nnet_unimplemented_continue();
        return nullptr;
    }
    auto replaceMutator =
        ReplaceVariable(replace.oldIters.at(0), replace.psis.at(0));
    auto ret = replaceMutator(cur);
    return ret;
}

} // namespace nnet