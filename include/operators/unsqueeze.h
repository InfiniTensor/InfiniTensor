#pragma once

#include "core/operator.h"

namespace infini {
/**
 * @brief nsert single-dimensional entries to the shape of an input tensor.
 *
 */
class UnsqueezeObj : public OperatorObj {
    Shape axes;

  public:
    /**
     * @brief Construct a new Unsqueeze object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param axes List of integers indicating the dimensions to be inserted.
     */
    UnsqueezeObj(GraphObj *graph, Tensor input, Tensor output, Shape axes);
    OP_CLONE(UnsqueezeObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    inline Shape getAxes() const { return axes; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
