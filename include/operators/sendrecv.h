#pragma once
#include "core/operator.h"

namespace infini {
/**
 *
 * https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2193/user-guide/docs/index.html
 */
class SendRecvObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new SendRecv object
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input send input
     * @param output recv output
     * @param source the send rank
     * @param destination the recv rank
     * @param dims The shape of the output tensor.
     */
    SendRecvObj(GraphObj *graph, Tensor input, Tensor output, int source,
                int destination, Shape dims);
    OP_CLONE(SendRecvObj);

    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;

    int getSource() const { return source; }
    int getDestination() const { return destination; }
    inline Shape getShape() const { return dims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  protected:
    int source;
    int destination;
    Shape dims;
};
} // namespace infini
