#pragma once
#include "core/operator.h"

namespace infini {
class MotionObj : public OperatorObj {
    bool useQemu;

  public:
    MotionObj(GraphObj *graph, OpType type, TensorVec values, bool useQemu);
    OP_CLONE(MotionObj);
    optional<vector<Shape>> inferShape(const TensorVec &values) override;
    vector<DataType> inferDataType(const TensorVec &) const override;
    string toString() const override;
    int numInputs() const override {
        return type == OpType::MotionFeatures ? 8 : 7;
    }
    int numOutputs() const override {
        return type == OpType::MotionFeatures ? 4 : 1;
    }
    bool executesQemu() const { return useQemu; }

  private:
    vector<int> getOpAttrVector() const override;
    vector<int> getWorkloadVector() const override;
};
} // namespace infini
