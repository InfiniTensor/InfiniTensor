#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Compute the reduction of input tensor's elements along certain axes.
 *
 */
class ReduceBaseObj : public OperatorObj {
  protected:
    set<int> axes; // axis to reduce
    bool keepDims;

  public:
    /**
     * @brief Construct a new Reduce object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param opType The operation type. Should be a Reduce operation.
     * @param input The input tensor.
     * @param output The output tensor.
     * @param axes Axes to reduce.
     * @param keepDims Keep the reduced dimensions or not.
     */
    ReduceBaseObj(GraphObj *graph, OpType opType, Tensor input, Tensor output,
                  const optional<vector<int>> &axes, bool keepDims);
    ~ReduceBaseObj() override {
      if (opDesc) {
          try {
              if ( type == OpType::ReduceMax ){
                CHECK_ERROR(infiniopDestroyReduceMaxDescriptor(
                  (infiniopReduceMaxDescriptor_t)opDesc));
                
              }else if(type == OpType::ReduceMean){
                CHECK_ERROR(infiniopDestroyReduceMeanDescriptor(
                  (infiniopReduceMeanDescriptor_t)opDesc));
              }else if(type == OpType::ReduceMin)
              {
                  CHECK_ERROR(infiniopDestroyReduceMinDescriptor(
                      (infiniopReduceMinDescriptor_t)opDesc));
              } else {
                  IT_ASSERT(false, "Unsupported reduce operator type "
                                    "for infini op destroy");
              }
          } catch (const std::exception &e) {
              std::cerr << "Error in ~ReduceBaseObj: " << e.what() << std::endl;
          }
      }
  }
  void initInfiniOp(const Runtime context) override;
                
    OP_CLONE(ReduceBaseObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    bool isReduced(int idx) const;
    const set<int> &getAxes() const { return axes; }
    bool getKeepDims() const { return keepDims; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class ReduceMeanObj : public ReduceBaseObj {
  public:
    ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                  const optional<vector<int>> &axes, bool keepDims = true);
};
class ReduceMaxObj : public ReduceBaseObj {
  public:
    ReduceMaxObj(GraphObj *graph, Tensor input, Tensor output,
                 const optional<vector<int>> &axes, bool keepDims = true);
};

class ReduceMinObj : public ReduceBaseObj {
  public:
    ReduceMinObj(GraphObj *graph, Tensor input, Tensor output,
                 const optional<vector<int>> &axes, bool keepDims = true);
};


class ReduceSumObj : public ReduceBaseObj {
  public:
    ReduceSumObj(GraphObj *graph, Tensor input, Tensor output,
                 const optional<vector<int>> &axes, bool keepDims = true);
};
} // namespace infini
