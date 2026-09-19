#pragma once

#include "core/operator.h"

namespace infini {
/**
 * @brief Repeat the input tensor a given number of times along each dimension.
 *
 */
class TileObj : public OperatorObj {
    Shape repeats;

  public:
    /**
     * @brief Construct a new Tile object whose repeat counts are a constant.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param repeats How many times to repeat each dimension of the input. One
     * per input dimension, in the input's order.
     */
    TileObj(GraphObj *graph, Tensor input, Tensor output, Shape repeats);
    /**
     * @brief Construct a new Tile object reading its repeat counts from an edge
     * of the graph rather than from a constant.
     *
     * This is the shape ONNX Tile takes, whose second input carries the counts.
     * Under a dynamic input those counts are usually worked out from the shape
     * of another tensor, so the model holds no constant to read. They have to
     * be known while shapes are inferred, which holds when they describe
     * dimensions rather than data -- see `TensorObj::getShapeValue`.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param repeats The tensor holding the repeat counts.
     * @param output The output tensor.
     */
    TileObj(GraphObj *graph, Tensor input, Tensor repeats, Tensor output);
    OP_CLONE(TileObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DimSource> dimSources(size_t output, size_t dim) const override;

    std::string toString() const override;
    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }

    /// @brief The repeat counts as last read, one per input dimension.
    inline Shape getRepeats() const { return repeats; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

} // namespace infini
