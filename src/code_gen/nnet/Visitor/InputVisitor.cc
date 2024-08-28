#include "nnet/Visitor/InputVisitor.h"

namespace nnet {

void InputVisitor::visit_(const Tensor &c) { inputs.emplace_back(c); }

} // namespace nnet