#include "code_gen/nnet/Pass/Rule5RangeRelaxation.h"
#include "code_gen/nnet/Visitor/RangeRelaxFunctor.h"

namespace nnet {

void Rule5RangeRelaxation::transform(Formula &origin, int depth, Expr &rCur) {
    rule5RangeRelaxation(origin, depth, rCur);
}

Expr Rule5RangeRelaxation::rule5RangeRelaxation(Formula &origin, int depth,
                                                Expr &rCur) {
    auto cur = as<RangeOpNode>(rCur);
    if (cur->hasPaddings()) {
        string msg = "====== END rule5RangeRelaxation: Paddings exist \n";
        dbg(msg);
        return nullptr;
    }

    // Infer meaningful calculation range
    RangeRelaxFunctor rangeRexlaxtionFunctor{cur};
    RangeMap rangeMap = rangeRexlaxtionFunctor(cur);
    auto relaxedCur = make_ref<RangeOpNode>(*cur);
    bool isRelaxed = false;
    vector<int> paddings;
    // check whether narrow the calculation range
    for (size_t i = 0; i < cur->getLoopVarRanges().size(); ++i) {
        const auto &[iter, iterRange] =
            cur->getVarRange(IterationType::Loop, i);
        if (auto it = rangeMap.find(iter); it != rangeMap.end()) {
            // intersection of validRange and iterRange is necessary computation
            // TODO: it is redundant with RangeRelaxFunctor::intersectRangeMaps.
            // An independent Range class might be necessary.
            const Range &validRange = it->second;
            Range relaxedRange{max(iterRange.first, validRange.first),
                               min(iterRange.second, validRange.second)};
            if (relaxedRange != iterRange) {
                isRelaxed = true;
                relaxedCur->setVarRange(IterationType::Loop, i,
                                        {iter, relaxedRange});
                paddings.emplace_back(
                    max(relaxedRange.first - iterRange.first,
                        iterRange.second - relaxedRange.second));
            } else
                paddings.emplace_back(0);
        } else
            paddings.emplace_back(0);
    }
    relaxedCur->setPaddings(paddings);
    if (!isRelaxed) {
        string msg = "====== END rule5RangeRelaxation: Relaxation not found\n";
        dbg(msg);
        return nullptr;
    }

    // next searching step
    string msg = "====== END rule5RangeRelaxation: relax iterating ranges ";
    string detailedMsg;
    for (size_t i = 0; i < cur->getLoopVarRanges().size(); ++i) {
        const auto &[v, a] = cur->getVarRange(IterationType::Loop, i);
        const auto &[_, b] = relaxedCur->getVarRange(IterationType::Loop, i);
        if (a != b) {
            detailedMsg += v->getName();
            detailedMsg +=
                " (" + to_string(a.first) + "," + to_string(a.second) + ") to";
            detailedMsg +=
                " (" + to_string(b.first) + "," + to_string(b.second) + "),";
        }
    }
    msg += detailedMsg + "\n";
    dbg(msg);
    nextStep(origin, depth, rCur, relaxedCur, detailedMsg);
    return relaxedCur;
}

} // namespace nnet