#pragma once
#include "core/operator.h"

namespace infini {
/**
 *
 * https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2193/user-guide/docs/index.html
 */
class SendObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new SendRecv object
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input send input
     * @param output recv output
     * @param source the send rank
     * @param destination the recv rank
     */
    SendObj(GraphObj *graph, Tensor input, int source, int destination,
            Tensor output = nullptr);
    OP_CLONE(SendObj);

    int numInputs() const override { return 1; }
    int numOutputs() const override { return outputs.size(); }
    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    int getSourceRank() const { return source; }
    int getDestinationRank() const { return destination; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  protected:
    int source;
    int destination;
};
} // namespace infini
