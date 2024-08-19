#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class CountRoutineVisitor : public ExprTreeVisitor {
  private:
    vector<int> cnts;

  public:
    CountRoutineVisitor(int _verobse = 0)
        : ExprTreeVisitor(1, 1, 1, 1, _verobse) {}
    void visit_(const Tensor &c) override;
    vector<int> count(const Expr &root);
    bool match(const Expr &root, int nMatmul = 0, int nConv = 0,
               int nElement = 0, int nSg2bmm = 0, int nLongformerGBMM = 0);
};
} // namespace nnet