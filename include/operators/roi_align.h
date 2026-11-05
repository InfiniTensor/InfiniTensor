#pragma once

#include "core/operator.h"
#include <tuple>

namespace infini {

class RoiAlignObj : public OperatorObj {
  public:
    RoiAlignObj(GraphObj *graph, Tensor data, Tensor boxes, Tensor output,
                 int height, int width, int samplingRatio, float spatialScale, bool aligned);
    OP_CLONE(RoiAlignObj);
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }
    auto getAttributes() const { return std::tuple(samplingRatio, spatialScale, aligned); }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    int height, width, samplingRatio;
    float spatialScale;
    bool aligned;
};

} // namespace infini
