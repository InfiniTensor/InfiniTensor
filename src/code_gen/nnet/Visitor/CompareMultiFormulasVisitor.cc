#include "code_gen/nnet/Visitor/CompareMultiFormulasVisitor.h"

namespace nnet {

bool CompareMultiFormulasVisitor::compare(const VecExpr &roots) {
    if (roots.empty())
        return false;
    vector<RangeOp> rangeOps;
    for (const auto &root : roots) {
        if (auto rangeOp = as<RangeOpNode>(root))
            rangeOps.emplace_back(rangeOp);
        else
            return false;
    }
    const auto pattern = rangeOps[0];
    for (auto rangeOp : rangeOps) {
        if (pattern->getNumOutputDims() != rangeOp->getNumOutputDims()) {
            return false;
        }
        for (int i = 0; i < pattern->getNumOutputDims(); ++i)
            if (pattern->getVarRange(0, i).second !=
                rangeOp->getVarRange(0, i).second) {
                return false;
            }
        for (size_t i = 0; i < pattern->getSumVarRanges().size(); ++i)
            if (pattern->getVarRange(1, i).second !=
                rangeOp->getVarRange(1, i).second) {
                return false;
            }
    }
    return true;
}

} // namespace nnet