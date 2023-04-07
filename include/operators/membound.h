#pragma once
#include "core/operator.h"
#include "nnet/expr.h"

namespace infini {

class MemBoundObj : public OperatorObj {
  private:
    std::vector<nnet::Tensor> nnetInputs;
    nnet::Expr expr;
    double exec_time;
    std::string hint;
    HashType hash;
    int n, f, h, w;

  public:
    MemBoundObj(GraphObj *graph, const TensorVec &input,
                const TensorVec &output,
                const std::vector<nnet::Tensor> &nnetInputs, nnet::Expr expr,
                double exec_time, std::string hint = {});
    OP_CLONE(MemBoundObj);

    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return outputs.size(); }
    const vector<nnet::Tensor> &getNnetInputs() const { return nnetInputs; }
    const nnet::Expr getNnetExpr() const { return expr; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    HashType getHash() const;
};

} // namespace infini
