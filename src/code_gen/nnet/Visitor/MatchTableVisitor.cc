#include "code_gen/nnet/Visitor/MatchTableVisitor.h"

namespace nnet {

void MatchTableVisitor::visit_(const BinaryOp &c, const Tensor &tensor, int dim,
                               optional<int> stride) {
    if (c->getOpType() == OpType::Add) {
        dispatch(c->getLhs(), tensor, dim, stride);
        dispatch(c->getRhs(), tensor, dim, stride);
    } else if (c->getOpType() == OpType::Sub) {
        dispatch(c->getLhs(), tensor, dim, stride);
        if (stride)
            *stride = -*stride;
        dispatch(c->getRhs(), tensor, dim, stride);
    } else if (c->getOpType() == OpType::Mul) {
        const optional<int> &lStride = subexprStride[c->getLhs().get()];
        const optional<int> &rStride = subexprStride[c->getRhs().get()];
        optional<int> lCurStride =
            (stride && rStride) ? optional(*stride * *rStride) : nullopt;
        optional<int> rCurStride =
            (stride && lStride) ? optional(*stride * *lStride) : nullopt;
        dispatch(c->getLhs(), tensor, dim, lCurStride);
        dispatch(c->getRhs(), tensor, dim, rCurStride);
    } else {
        hasUnsupportedOp = true;
    }
}

void MatchTableVisitor::visit_(const Subscript &c, const Tensor &tensor,
                               [[maybe_unused]] int dim,
                               [[maybe_unused]] optional<int> stride) {
    assert(!tensor); // Should not be set until visit a tensor
    auto object = as<TensorNode>(c->getObject());
    assert(object);
    tensors.emplace_back(object);
    int currentStride = 1;
    for (int i = (int)c->getDims() - 1; i >= 0; --i) {
        this->dispatch(c->getIndex(i), object, i, currentStride);
        currentStride *= object->getShape(i);
    }
}
void MatchTableVisitor::visit_(const Var &c, const Tensor &tensor, int dim,
                               optional<int> stride) {
    appearance.try_emplace(c);
    appearance[c].emplace_back(pair(tensor, dim));
    strideTable[c].emplace_back(tensor.get(), dim, stride.value());
}

void MatchTableVisitor::visit_([[maybe_unused]] const Constant &c,
                               [[maybe_unused]] const Tensor &tensor,
                               [[maybe_unused]] int dim,
                               [[maybe_unused]] optional<int> stride) {
    return;
}

} // namespace nnet