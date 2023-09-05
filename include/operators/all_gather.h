#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief The AllGather operation gathers N values from k ranks into
 * an output of size k*N, and distributes that result to all ranks.
 * The output is ordered by rank index.
 *
 * For more details:
 * https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather
 */
class AllGatherObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new AllGather object
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor from this rank.
     * @param outputs A list of output tensors collected from all ranks.
     * @param world_size Total number of ranks.
     */
    AllGatherObj(GraphObj *graph, Tensor input, std::optional<TensorVec>,
                 int world_size);
    OP_CLONE(AllGatherObj);

    int numInputs() const override { return 1; }
    int numOutputs() const override { return world_size; }
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;

    int getWorldSize() const { return world_size; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  protected:
    int world_size;
};
} // namespace infini
