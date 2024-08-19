#include "code_gen/nnet/Pass/Rule9RangeMagnify.h"
#include "code_gen/nnet/Visitor/RangeMagnifyVisitor.h"

namespace nnet {

void Rule9RangeMagnify::transform(Formula &origin, int depth, Expr &rCur) {
    auto cur = as<RangeOpNode>(rCur);
    if (cur->hasPaddings()) {
        string msg = "====== END rule9RangeMagnify: Paddings exist \n";
        dbg(msg);
        return;
    }
    // HACK for conv5x5
    vector<VarRangePair> newSumVarRanges;
    for (const auto &[var, range] : cur->getSumVarRanges()) {
        if (range.first == 0 && range.second == 5) {
            newSumVarRanges.emplace_back(
                var, Range{range.first, (range.second + 2) / 3 * 3});
        } else
            newSumVarRanges.emplace_back(var, range);
    }
    if (newSumVarRanges.empty())
        return;
    auto magnifiedCur = RangeMagnifyVisitor().magnify(cur, newSumVarRanges);
    if (!magnifiedCur)
        return;

    // next searching step
    string msg = "====== END rule9RangeMagnify: relax iterating ranges ";
    for (size_t i = 0; i < cur->getSumVarRanges().size(); ++i) {
        const auto &[v1, a] = cur->getVarRange(IterationType::Sum, i);
        const auto &[v2, b] = magnifiedCur->getVarRange(IterationType::Sum, i);
        assert(v1->getName() == v2->getName());
        if (a != b) {
            msg += v1->getName();
            msg +=
                " (" + to_string(a.first) + "," + to_string(a.second) + ") to";
            msg += " (" + to_string(b.first) + "," + to_string(b.second) + "),";
        }
    }
    msg += "\n";
    dbg(msg);
    nextStep(origin, depth, rCur, magnifiedCur);
    return;
}

} // namespace nnet