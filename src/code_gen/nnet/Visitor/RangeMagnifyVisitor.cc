#include "code_gen/nnet/Visitor/RangeMagnifyVisitor.h"
#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"

namespace nnet {

Expr RangeMagnifyVisitor::visit_(const RangeOp &c) {
    if (auto expr = Mutator::visit_(c)) {
        auto ret = as<RangeOpNode>(expr);
        ret->setSumIterator(newSumVarRanges);
        return ret;
    } else
        return nullptr;
}

Expr RangeMagnifyVisitor::visit_(const Subscript &c) {
    auto tensor = as<TensorNode>(c->getObject());
    if (!tensor)
        return nullptr;
    // Check new ranges
    bool paddingMagnify = false;
    vector<Range> tensorRanges = c->getObjectRanges();
    vector<int> paddingsDelta(tensorRanges.size(), 0);
    for (int i = 0; i < (int)c->getDims(); ++i) {
        auto indexRange =
            SimplifyExprVisitor().getExprRange(c->getIndex(i), newRangeOp);
        if (!indexRange.has_value())
            return nullptr;
        int delta = max(tensorRanges[i].first - indexRange->first,
                        indexRange->second - tensorRanges[i].second);
        if (delta > 0) {
            paddingMagnify = true;
            paddingsDelta[i] = delta;
        }
    }
    if (!paddingMagnify)
        return nullptr;
    // Create new tensor. Direct add paddings to the Tensor.
    auto newTensor = make_ref<TensorNode>(*tensor);
    for (int i = 0; i < newTensor->getDims(); ++i)
        newTensor->setPadding(i, newTensor->getPadding(i) + paddingsDelta[i]);
    auto newSub = make_ref<SubscriptNode>(*c);
    newSub->setObject(newTensor);
    return newSub;
}

RangeOp
RangeMagnifyVisitor::magnify(const RangeOp &root,
                             const vector<VarRangePair> &_newSumVarRanges) {
    newSumVarRanges = _newSumVarRanges;
    newRangeOp = make_ref<RangeOpNode>(*root);
    newRangeOp->setSumIterator(newSumVarRanges);
    const auto &newCur = as<RangeOpNode>(dispatch(root));
    return newCur;
}

} // namespace nnet