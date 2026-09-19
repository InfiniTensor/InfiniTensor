#pragma once
#include "core/operator.h"

namespace infini {
/**
 *  @brief Broadcast the input tensor following the given shape and the
 * broadcast rule.
 *
 */
class ExpandObj : public OperatorObj {
    Shape dims;

  public:
    /**
     * @brief Construct a new Expand object.
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param dims The shape you want to expand to, following the broadcast
     * rule.
     */
    ExpandObj(GraphObj *graph, Tensor input, Tensor output, Shape dims);
    /**
     * @brief Construct a new Expand object whose target is an edge of the graph
     * rather than a constant.
     *
     * An exporter writes the target of an expand over a dynamic input as a
     * computation on that input's shape, so the model holds no constant to
     * read: the target is only a number once a shape is given. `shape` carries
     * it as a shape value, which `inferShape` reads afresh each time.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param shape The rank one tensor holding the shape to expand to.
     * @param output The output tensor.
     */
    ExpandObj(GraphObj *graph, Tensor input, Tensor shape, Tensor output);
    OP_CLONE(ExpandObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    /// A dimension the target asks more than one of comes out at that number
    /// whatever the input holds there, because broadcasting either agrees with
    /// the target or stretches a one up to it. A target of one leaves the input
    /// dimension as it is, so that one is followed.
    vector<DimSource> dimSources(size_t output, size_t dim) const override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    /// The target as it was asked for, which is not what comes out: an input
    /// dimension wider than the target's keeps its own size. Read the output's
    /// shape for that.
    Shape getShape() const { return dims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
