#include "nnet/Visitor/MatchReshapeVisitor.h"
#include "nnet/Visitor/MergeMemboundMutator.h"
#include "nnet/Visitor/SimplifyExprVisitor.h"

namespace nnet {

bool MatchReshapeVisitor::visit_(const RangeOp &memboundRangeOp) {
    // Merge nested stages
    auto rangeOp =
        as<RangeOpNode>(MergeMemboundMutator({memboundRangeOp}).merge());
    assert(rangeOp);
    auto sub = as<SubscriptNode>(rangeOp->getSummand());
    if (!sub)
        return false;
    auto sumRanges = rangeOp->getSumVarRanges();
    for (auto const &[var, range] : sumRanges) {
        if (range.second - range.first != 1)
            return false;
    }

    const auto objectRanges = sub->getObjectRangesWithoutPaddings();
    const auto indices = sub->getIndex();
    Expr indexExpr;
    int stride = 1;
    for (int i = indices.size() - 1; i >= 0; --i) {
        indexExpr = indexExpr + indices.at(i) * stride;
        stride *= (objectRanges.at(i).second - objectRanges.at(i).first);
    }

    SimplifyExprVisitor simplifyExprVisitor;
    simplifyExprVisitor.simplify(indexExpr);
    auto exprStrides = simplifyExprVisitor.getStrides();

    auto varRanges = rangeOp->getLoopVarRanges();
    stride = 1;
    // compare strides of variables in RangeOP and index
    for (auto i = varRanges.rbegin(); i != varRanges.rend(); ++i) {
        const bool alwaysZero = i->second.first == 0 && i->second.second == 1;
        if (!alwaysZero && exprStrides[i->first] != stride)
            return false;
        stride *= (i->second.second - i->second.first);
    }
    return true;
}

} // namespace nnet