#pragma once
#include "core/operator.h"

namespace infini {
/**
 *
 * https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2193/user-guide/docs/index.html
 */
class RecvObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new SendRecv object
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input default nullptr, because recv does not have input.
     * @param output recv output
     * @param source the send rank
     * @param destination the recv rank
     * @param dims The shape of the output tensor.
     */
    RecvObj(GraphObj *graph, Tensor output, int source, int destination,
            Shape dims, int outputType, Tensor input = nullptr);
    OP_CLONE(RecvObj);

    int numInputs() const override { return inputs.size(); }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    DataType getDType() const;
    int getSourceRank() const { return source; }
    int getDestinationRank() const { return destination; }
    inline Shape getShape() const { return dims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  protected:
    int source;
    int destination;
    Shape dims;
    int outputType;
};
} // namespace infini
