#pragma once

#include "core/operator.h"

namespace infini {

/**
 * @brief Remove single-dimensional entries from the shape of a tensor.
 *
 */
class SqueezeObj : public OperatorObj {
    Shape axes;

  public:
    /**
     * @brief Construct a new Squeeze object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param axes List of integers indicating the dimensions to squeeze.
     */
    SqueezeObj(GraphObj *graph, Tensor input, Tensor output, Shape axes);
    OP_CLONE(SqueezeObj);

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
