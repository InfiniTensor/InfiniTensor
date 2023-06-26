#pragma once
#include "core/operator.h"

namespace infini {
class MemBoundObj : public OperatorObj {
  public:
    MemBoundObj(GraphObj *graph, const TensorVec &input,
                const TensorVec &output,
                const std::vector<nnet::Tensor> &nnetInputs, nnet::Expr expr,
                double exec_time, std::string hint = {}) {}
};

} // namespace infini
