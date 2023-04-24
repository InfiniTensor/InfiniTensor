#pragma once

#include "../graph.h"

namespace optimization {

class Conv {
    Operator const &op;

  public:
    explicit Conv(Operator &op) : op(op) {}
    explicit Conv(Operator const &op) : op(op) {}

    Arc<Tensor> const &input() const { return op.inputs.at(0); }
    Arc<Tensor> const &weight() const { return op.inputs.at(1); }
    Arc<Tensor> const &delations() const { return op.inputs.at(2); }
    Arc<Tensor> const &pads() const { return op.inputs.at(3); }
    Arc<Tensor> const &strides() const { return op.inputs.at(4); }
    Arc<Tensor> const &output() const { return op.outputs.at(0); }
};

} // namespace optimization
