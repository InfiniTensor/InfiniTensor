#pragma once
#include "core/operator.h"

namespace infini {
class AllReduceBaseObj : public OperatorObj {

  public:
    AllReduceBaseObj(GraphObj *graph, OpType opType, Tensor input,
                     Tensor output);
    OP_CLONE(AllReduceBaseObj);
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override {
        return {{inputs[0]->getDims()}};
    };
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override {
        return {inputs[0]->getDType()};
    };
};

class AllReduceSumObj : public AllReduceBaseObj {
  public:
    AllReduceSumObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceProdObj : public AllReduceBaseObj {
  public:
    AllReduceProdObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceMinObj : public AllReduceBaseObj {
  public:
    AllReduceMinObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceMaxObj : public AllReduceBaseObj {
  public:
    AllReduceMaxObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceAvgObj : public AllReduceBaseObj {
  public:
    AllReduceAvgObj(GraphObj *graph, Tensor input, Tensor output);
};

} // namespace infini
