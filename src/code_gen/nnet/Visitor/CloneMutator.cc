#include "code_gen/nnet/Visitor/CloneMutator.h"

namespace nnet {

Expr CloneMutator::visit_(const Constant &c) { return c; }
Expr CloneMutator::visit_(const Var &c) { return c; }
Expr CloneMutator::visit_(const Tensor &c) { return c; }

} // namespace nnet