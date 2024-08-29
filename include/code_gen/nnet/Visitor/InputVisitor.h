#pragma once
#include "code_gen/nnet/visitor.h"

namespace nnet {

class InputVisitor : public ExprTreeVisitor {
    vector<Tensor> inputs;

  public:
    int nInputs = 0;
    InputVisitor(int _verobse = 0) : ExprTreeVisitor(1, 1, 1, 0, _verobse) {}
    void visit_(const Tensor &c) override;

    /**
     * @brief Get the all inputs in the netsed stages
     */
    vector<Tensor> getInputs(const RangeOp &_rangeOp) {
        dispatch(_rangeOp);
        return inputs;
    }
};

} // namespace nnet