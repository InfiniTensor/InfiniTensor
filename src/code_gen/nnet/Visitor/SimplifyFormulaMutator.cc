#include "code_gen/nnet/Visitor/SimplifyFormulaMutator.h"
#include "code_gen/nnet/Visitor/SimplifyExprVisitor.h"

namespace nnet {

Expr SimplifyFormulaMutator::visit_(const Subscript &c) {
    ++nSubscripts;
    if (verbose)
        dbg(*c);
    bool modified = false;
    auto ret = make_ref<SubscriptNode>(*c);
    for (size_t i = 0; i < ret->getDims(); ++i) {
        const auto &e = ret->getIndex(i);
        if (const auto &mutated = SimplifyExprVisitor().simplify(e)) {
            modified = true;
            ret->setIndex(i, mutated);
        }
    }
    return (modified) ? ret : nullptr;
}

Expr SimplifyFormulaMutator::simplify(const Expr &expr) {
    nSubscripts = 0;
    const auto &ret = dispatch(expr);
    nnet_assert(nSubscripts > 0,
                "Subscript NOT found. Use SimplifyFormulaMutator?");
    return (ret) ? ret : expr;
}

} // namespace nnet