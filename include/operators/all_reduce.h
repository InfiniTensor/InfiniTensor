#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief The AllReduce operation is performing reductions on data (sum, min,
 * max, avg, or div) across devices and writing the result in the
 * receive buffers of every rank. For example, in an allreduce operation between
 * k ranks and performing a sum, each rank will provide an array Vk of N values,
 * and receive an identical arrays S of N values, where S[i] =
 * V0[i]+V1[i]+…+Vk-1[i].
 *
 * For more details:
 * https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce
 */
class AllReduceBaseObj : public OperatorObj {

  public:
    /**
     * @brief Construct a new AllReduce base object. Should be called by every
     * child class constructor, but not directly.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param opType The operation type. This param is taken care of by child
     * classes.
     * @param input The input tensor from this rank.
     * @param output The output tensor, same size as input.
     */
    AllReduceBaseObj(GraphObj *graph, OpType opType, Tensor input,
                     Tensor output);
    OP_CLONE(AllReduceBaseObj);

    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override {
        return {{inputs[0]->getDims()}};
    };

    std::string toString() const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override {
        return {inputs[0]->getDType()};
    };
};

class AllReduceSumObj : public AllReduceBaseObj {
  public:
    AllReduceSumObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceProdObj : public AllReduceBaseObj {
  public:
    AllReduceProdObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceMinObj : public AllReduceBaseObj {
  public:
    AllReduceMinObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceMaxObj : public AllReduceBaseObj {
  public:
    AllReduceMaxObj(GraphObj *graph, Tensor input, Tensor output);
};

class AllReduceAvgObj : public AllReduceBaseObj {
  public:
    AllReduceAvgObj(GraphObj *graph, Tensor input, Tensor output);
};

} // namespace infini
