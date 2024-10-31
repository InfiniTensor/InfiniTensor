#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief The base class for GlobalAvgPool and GlobalMaxPool.
 *
 */
class GlobalPoolObj : public OperatorObj {
  private:
    int n, c;

  public:
    /**
     * @brief Construct a new GlobalPool object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param optype Operator type of this pooling operator.
     * @param input The input tensor.
     * @param output The output tensor.
     * shape.
     */
    GlobalPoolObj(GraphObj *graph, OpType optype, Tensor input, Tensor output);
    OP_CLONE(GlobalPoolObj);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class GlobalMaxPoolObj : public GlobalPoolObj {
  public:
    GlobalMaxPoolObj(GraphObj *graph, Tensor input, Tensor output)
        : GlobalPoolObj(graph, OpType::GlobalMaxPool, input, output) {}
};
class GlobalAvgPoolObj : public GlobalPoolObj {
  public:
    GlobalAvgPoolObj(GraphObj *graph, Tensor input, Tensor output)
        : GlobalPoolObj(graph, OpType::GlobalAveragePool, input, output) {}
};
}; // namespace infini
