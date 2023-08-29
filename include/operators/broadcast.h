#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief The Broadcast operation copies an N-element buffer on the root rank to
 * all ranks.
 *
 * For more details:
 * https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast
 */
class BroadcastObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new Broadcast object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor. Only root needs to initialize it with
     * data.
     * @param output The output tensor, same size as input.
     * @param root The root rank who performs the broadcast.
     */
    BroadcastObj(GraphObj *graph, Tensor input, Tensor output, int root);
    OP_CLONE(BroadcastObj);

    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override {
        return {{inputs[0]->getDims()}};
    };

    std::string toString() const override;

    int getRoot() const { return root; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override {
        return {inputs[0]->getDType()};
    };

  protected:
    // The rank who broadcasts data among this communication group
    int root;
};

} // namespace infini
