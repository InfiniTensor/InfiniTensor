#include "code_gen/nnet/Visitor/ReplaceNodeMutator.h"

namespace nnet {

Expr ReplaceNodeMutator::visit_(const Subscript &c) {
    if (c.get() == target)
        return replacement;
    return Mutator::visit_(c);
}
Expr ReplaceNodeMutator::visit_(const Tensor &c) {
    if (c.get() == target)
        return replacement;
    return nullptr;
}

Expr ReplaceNodeMutator::replace(const Expr &root, ExprNode *_target,
                                 const Expr &_replace) {
    target = _target;
    replacement = _replace;
    return dispatch(root);
}

} // namespace nnet