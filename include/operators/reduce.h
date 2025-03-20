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
    virtual ~ReduceBaseObj() {}
    OP_CLONE(ReduceBaseObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

    bool isReduced(int idx) const;
    const set<int> &getAxes() const { return axes; }
    bool getKeepDims() const { return keepDims; }

    void initInfiniOp(const Runtime context) override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class ReduceMeanObj : public ReduceBaseObj {
  public:
    ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                  const optional<vector<int>> &axes, bool keepDims = true)
        : ReduceBaseObj(graph, OpType::ReduceMean, input, output, axes, keepDims) {}
    ~ReduceMeanObj() override {
        if (opDesc) {
            try {
                CHECK_ERROR(infiniopDestroyReducemeanDescriptor(
                    (infiniopReducemeanDescriptor_t)opDesc));
            } catch (const std::exception &e) {
                std::cerr << "Error in ~ReduceMeanObj: " << e.what() << std::endl;
            }
        }
    }
};

class ReduceSumObj : public ReduceBaseObj {
  public:
    ReduceSumObj(GraphObj *graph, Tensor input, Tensor output,
                 const optional<vector<int>> &axes, bool keepDims = true)
        : ReduceBaseObj(graph, OpType::ReduceSum, input, output, axes, keepDims) {}      
};

class ReduceMaxObj : public ReduceBaseObj {
  public:
    ReduceMaxObj(GraphObj *graph, Tensor input, Tensor output,
                 const optional<vector<int>> &axes, bool keepDims = true)
        : ReduceBaseObj(graph, OpType::ReduceMax, input, output, axes, keepDims) {}
    ~ReduceMaxObj() override {
        if (opDesc) {
            try {
                CHECK_ERROR(infiniopDestroyReducemaxDescriptor(
                    (infiniopReducemaxDescriptor_t)opDesc));
            } catch (const std::exception &e) {
                std::cerr << "Error in ~ReduceMaxObj: " << e.what() << std::endl;
            }
        }
    }
};

class ReduceMinObj : public ReduceBaseObj {
  public:
    ReduceMinObj(GraphObj *graph, Tensor input, Tensor output,
                 const optional<vector<int>> &axes, bool keepDims = true)
        : ReduceBaseObj(graph, OpType::ReduceMin, input, output, axes, keepDims) {}
    ~ReduceMinObj() override {
        if (opDesc) {
            try {
                CHECK_ERROR(infiniopDestroyReduceminDescriptor(
                    (infiniopReduceminDescriptor_t)opDesc));
            } catch (const std::exception &e) {
                std::cerr << "Error in ~ReduceMinObj: " << e.what() << std::endl;
            }
        }
    }
};
} // namespace infini
