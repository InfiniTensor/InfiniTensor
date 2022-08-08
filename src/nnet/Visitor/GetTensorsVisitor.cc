#include "nnet/Visitor/GetTensorsVisitor.h"

namespace nnet {

void GetTensorsVisitor::visit_(const Tensor &c) {
    tensors.try_emplace(c->getName(), c);
}

} // namespace nnet