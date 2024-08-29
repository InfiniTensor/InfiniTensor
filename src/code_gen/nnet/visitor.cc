#include "code_gen/nnet/visitor.h"
namespace nnet {

Expr Mutator::visit_([[maybe_unused]] const Constant &c) { return nullptr; }

Expr Mutator::visit_(const BinaryOp &c) {
    if (verbose)
        dbg(*c);
    bool modified = false;
    auto ret = make_ref<BinaryOpNode>(*c);
    if (auto e = this->dispatch(ret->getLhs()); e) {
        modified = true;
        ret->setLhs(e);
    }
    if (auto e = this->dispatch(ret->getRhs()); e) {
        modified = true;
        ret->setRhs(e);
    }
    return (modified) ? ret : nullptr;
}

Expr Mutator::visit_(const RangeOp &c) {
    if (verbose)
        dbg(*c);
    bool modified = false;
    auto ret = make_ref<RangeOpNode>(*c);
    if (auto mutated = this->dispatch(ret->getSummand()); mutated) {
        modified = true;
        ret->setSummand(mutated);
    }
    // NOT visit iterators and its ranges
    return (modified) ? ret : nullptr;
}

Expr Mutator::visit_(const Subscript &c) {
    if (verbose)
        dbg(*c);
    bool modified = false;
    auto ret = make_ref<SubscriptNode>(*c);
    for (size_t i = 0; i < ret->getDims(); ++i) {
        const auto &e = ret->getIndex(i);
        if (const auto &mutated = this->dispatch(e); mutated) {
            modified = true;
            ret->setIndex(i, mutated);
        }
    }
    if (auto mutated = this->dispatch(ret->getObject()); mutated) {
        modified = true;
        ret->setObject(mutated);
    }
    return (modified) ? ret : nullptr;
}

Expr Mutator::visit_([[maybe_unused]] const Var &c) { return nullptr; }

Expr Mutator::visit_([[maybe_unused]] const Tensor &c) { return nullptr; }

Expr Mutator::visit_(const Func &c) {
    if (verbose)
        dbg(c);
    bool modified = false;
    auto ret = make_ref<FuncNode>(*c);
    if (auto mutated = dispatch(c->getObject())) {
        modified = true;
        ret->setObject(mutated);
    }
    return (modified) ? ret : nullptr;
}

void ExprTreeVisitor::visit_(const RangeOp &c) {
    if (inRange)
        dispatch(c->getSummand());
}
void ExprTreeVisitor::visit_(const BinaryOp &c) {
    if (inBinary) {
        dispatch(c->getLhs());
        dispatch(c->getRhs());
    }
}
void ExprTreeVisitor::visit_(const Subscript &c) {
    if (inSub) {
        dispatch(c->getObject());
        for (const auto &index : c->getIndex())
            dispatch(index);
    }
}
void ExprTreeVisitor::visit_([[maybe_unused]] const Var &c) {}
void ExprTreeVisitor::visit_([[maybe_unused]] const Constant &c) {}
void ExprTreeVisitor::visit_(const Tensor &c) {
    if (inTensor && c->getSource()) {
        if (const auto &expr = c->getSource()->getExpr(); expr)
            dispatch(expr);
    }
}
void ExprTreeVisitor::visit_(const Func &c) { dispatch(c->getObject()); }

} // namespace nnet
