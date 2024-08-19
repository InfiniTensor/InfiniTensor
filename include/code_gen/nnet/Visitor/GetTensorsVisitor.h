#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

// Get all tensors in the stage
class GetTensorsVisitor : public ExprTreeVisitor {
  private:
    unordered_map<string, Tensor> tensors;

    void visit_(const Tensor &c) override;

  public:
    GetTensorsVisitor(int _verobse = 0)
        : ExprTreeVisitor(1, 1, 1, 0, _verobse) {}
    auto get(const Expr &c) {
        dispatch(c);
        return tensors;
    }
};

} // namespace nnet