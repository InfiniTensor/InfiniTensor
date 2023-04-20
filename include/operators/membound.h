#pragma once
#include "core/operator.h"
#include "nnet/expr.h"

namespace infini {

class MemBoundObj : public OperatorObj {
  private:
    nnet::Expr expr;
    std::vector<nnet::Tensor>
        nnetInputs; // The order of inputs in nnetInputs should be consistant
                    // with inputs in infinitensor
    double exec_time;
    std::string hint;

    // Generated attributes
    HashType hash;
    nnet::Expr simplifiedExpr;
    HashType simplifiedHash;

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
    HashType getHash() const { return hash; }
    pair<const nnet::Expr, HashType> getSimplifiedNnetExpr() const {
        return {expr, hash};
    }
    double getEstimatedTime() const { return exec_time; }
    string toJson() const;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    static HashType calcHash(nnet::Expr expr);
    static bool checkOOB(nnet::Expr expr);
};

} // namespace infini
