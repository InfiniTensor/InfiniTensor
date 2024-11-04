#pragma once
#include "core/operator.h"

namespace infini {

class AscendPluginSubObj : public OperatorObj {
    int kernel_size;
    int stride;

  public:
    AscendPluginSubObj(GraphObj *graph, Tensor input, Tensor output,
                       int kernel_size, int stride);
    OP_CLONE(AscendPluginSubObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    int getKernelSize() const { return kernel_size; }
    int getStride() const { return stride; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
