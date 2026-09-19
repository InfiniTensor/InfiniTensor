#pragma once

#include "core/operator.h"

namespace infini {
/**
 * @brief Change the shape of the input tensor.
 *
 */
class ReshapeObj : public OperatorObj {
    Shape dims;
    Shape outputShape;

  public:
    /**
     * @brief Construct a new Reshape object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param dims The shape to infer the output shape.
     * @param outputShape The real shape of output tensor.
     */
    ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims);
    /**
     * @brief Construct a new Reshape object reading its target from a tensor.
     *
     * This is the shape of ONNX Reshape, whose second input carries the target.
     * That target has to be known while shapes are inferred, which holds when
     * it describes dimensions rather than data -- see
     * `TensorObj::getShapeValue`.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param shape The tensor holding the shape to infer the output shape.
     * @param output The output tensor.
     */
    ReshapeObj(GraphObj *graph, Tensor input, Tensor shape, Tensor output);
    OP_CLONE(ReshapeObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    /// Each output dimension follows whatever its own target element says of
    /// it, so the three kinds ONNX allows are told apart one element at a time:
    /// a positive number names the dimension and settles it, a zero keeps the
    /// input dimension in that position and follows exactly that one, and a
    /// minus one asks for what the others leave over, which the whole input
    /// shape decides. An element still moving claims nothing.
    vector<DimSource> dimSources(size_t output, size_t dim) const override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

    inline Shape getShape() const { return outputShape; }
    inline Shape getDims() const { return dims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

/**
 * @brief Reshape the input tensor into a one-dimensional tensor.
 * FIXME: Move to an independent file.
 * FIXME: Different parameter list with ONNX and Pytorch.
 *
 */
class FlattenObj : public OperatorObj {
    int axis;

  public:
    /**
     * @brief Construct a new Flatten object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output one-dimensional tensor.
     */
    FlattenObj(GraphObj *graph, Tensor input, Tensor output, int axis);
    OP_CLONE(FlattenObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    int getAxis() const { return axis; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

/**
 * @brief Copy the input tensor.
 * FIXME: Move to an independent file.
 *
 */
class IdentityObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new Identity object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor, which is the same as the input tensor.
     */
    IdentityObj(GraphObj *graph, Tensor input, Tensor output);
    OP_CLONE(IdentityObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    void inferShapeValue() override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
