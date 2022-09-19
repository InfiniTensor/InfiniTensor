#pragma once

#include "core/operator.h"

namespace infini {
class ReshapeObj : public OperatorObj {
    Shape dims;

  public:
    ReshapeObj(GraphObj *graph, Tensor input, Tensor output, const Shape &dims);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class FlattenObj : public OperatorObj {

  public:
    FlattenObj(GraphObj *graph, Tensor input, Tensor output);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class IdentityObj : public OperatorObj {

  public:
    IdentityObj(GraphObj *graph, Tensor input, Tensor output);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
