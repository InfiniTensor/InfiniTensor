#include "code_gen/nnet/Visitor/StrideVisitor.h"

namespace nnet {

optional<int> StrideVisitor::visit_(const Subscript &c) {
    if (verbose)
        dbg(*c);
    auto object = as<TensorNode>(c->getObject());
    assert(object);
    for (int i = (int)c->getDims() - 1; i >= 0; --i)
        this->dispatch(c->getIndex(i));
    return {};
}

optional<int> StrideVisitor::visit_(const Var &c) {
    if (verbose)
        dbg(*c);
    // assert(subexprStride.count(&c) == 0);
    subexprStride[c.get()] = {};
    return {};
}

optional<int> StrideVisitor::visit_(const Constant &c) {
    if (verbose)
        dbg(*c);
    optional ret{c->getValue()};
    // assert(subexprStride.count(&c) == 0);
    subexprStride[c.get()] = ret;
    return ret;
}

optional<int> StrideVisitor::visit_(const BinaryOp &c) {
    if (verbose)
        dbg(*c);
    optional<int> strideL = this->dispatch(c->getLhs());
    optional<int> strideR = this->dispatch(c->getRhs());
    if (!strideL || !strideR)
        return {};
    optional<int> ret;
    switch (c->getOpType()) {
    case OpType::Add:
        ret = optional(*strideL + *strideR);
        break;
    case OpType::Sub:
        ret = optional(*strideL - *strideR);
        break;
    case OpType::Mul:
        ret = optional(*strideL * *strideR);
        break;
    default:
        nnet_unimplemented_halt();
        break;
    }
    // assert(subexprStride.count(&c) == 0);
    subexprStride[c.get()] = ret;
    return ret;
}

} // namespace nnet