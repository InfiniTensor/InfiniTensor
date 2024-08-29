#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class FullPrinterVisitor : public ExprTreeVisitor {
  private:
    vector<tuple<string, Routine, Tensor>> q;

  public:
    FullPrinterVisitor(int _verobse = 0)
        : ExprTreeVisitor(1, 1, 1, 0, _verobse) {}
    void visit_(const Tensor &c) override;

    string print(const Expr &root);
    /**
     * @brief Get all tensors & OPs in a reversed order
     *
     * @param root
     * @return vector<<Output TensorName, RoutineNode, output tensor in NNet>>
     */
    const vector<tuple<string, Routine, Tensor>> &traverse(const Expr &root);
};

} // namespace nnet