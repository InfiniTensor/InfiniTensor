#pragma once
#include "core/operator.h"
#include "utils/lsm_sequence.h"
namespace infini {
class LSMSequenceObj : public OperatorObj {
    lsm::Parameters parameters;
    bool useQemu, traceAllSteps;

  public:
    LSMSequenceObj(GraphObj *graph, Tensor spikes, Tensor inputWeights,
                   Tensor recurrentWeights, Tensor stateIn,
                   const vector<int> &params, bool useQemu = true,
                   bool traceAllSteps = false);
    OP_CLONE(LSMSequenceObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;
    string toString() const override;
    int numInputs() const override { return 4; }
    int numOutputs() const override { return 3; }
    const lsm::Parameters &getParameters() const { return parameters; }
    bool executesQemu() const { return useQemu; }
    bool tracesAllSteps() const { return traceAllSteps; }

  private:
    vector<int> getOpAttrVector() const override;
    vector<int> getWorkloadVector() const override;
};

class MLPObj : public OperatorObj {
    bool useQemu;

  public:
    MLPObj(GraphObj *graph, Tensor x, Tensor w1, Tensor b1, Tensor w2,
           Tensor b2, bool useQemu = true);
    OP_CLONE(MLPObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &) const override;
    string toString() const override;
    int numInputs() const override { return 5; }
    int numOutputs() const override { return 1; }
    bool executesQemu() const { return useQemu; }

  private:
    vector<int> getOpAttrVector() const override;
    vector<int> getWorkloadVector() const override;
};
} // namespace infini
