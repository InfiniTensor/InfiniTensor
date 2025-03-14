#pragma once

#include "core/operator.h"

namespace infini {

class GatherBaseObj : public OperatorObj {
  protected:
    int axis;

  public:
    GatherBaseObj(OpType opType, TensorVec inputs, TensorVec outputs, int axis)
        : OperatorObj(opType, inputs, outputs), axis(axis) {}

    virtual ~GatherBaseObj() {}
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

    int getAxis() const { return axis; }
};

/**
 * @brief Gather and concatenate given positions on a certain dimension of the
 * input tensor using an index tensor.
 *
 */
class GatherObj : public GatherBaseObj {
  public:
    /**
     * @brief Construct a new Gather object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param indices The index tensor.
     * @param output The output tensor.
     * @param axis The axis to gather on.
     */
    GatherObj(GraphObj *graph, Tensor input, Tensor indices, Tensor output,
              int axis);
    OP_CLONE(GatherObj);
    ~GatherObj() override {
      if (opDesc) {
          try {
              if (type == OpType::Gather) {
                  CHECK_ERROR(infiniopDestroyGatherDescriptor(
                      (infiniopGatherDescriptor_t)opDesc));
              } else {
                  IT_ASSERT(false, "Unsupported gather operator type "
                                   "for infini op destroy");
              }
          } catch (const std::exception &e) {
              std::cerr << "Error in ~GatherObj: " << e.what()
                        << std::endl;
          }
      }
  }
    void initInfiniOp(const Runtime context) override;
    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  private:
    bool CheckIndexValid() const;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

/**
 * @brief GatherElements takes two inputs data and indices of the
 * same rank r >= 1 and an optional attribute axis that identifies
 * an axis of data.
 *
 */
class GatherElementsObj : public GatherBaseObj {
  public:
    /**
     * @brief Construct a new GatherElements object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param indices The index tensor.
     * @param output The output tensor. Same shape as indices.
     * @param axis The axis to gather on.
     */
    GatherElementsObj(GraphObj *graph, Tensor input, Tensor indices,
                      Tensor output, int axis);
    OP_CLONE(GatherElementsObj);
    ~GatherElementsObj() override {
      if (opDesc) {
          try {
              if (type == OpType::GatherElements) {
                  CHECK_ERROR(infiniopDestroyGatherElementsDescriptor(
                      (infiniopGatherElementsDescriptor_t)opDesc));
              } else {
                  IT_ASSERT(false, "Unsupported gather_elements operator type "
                                   "for infini op destroy");
              }
          } catch (const std::exception &e) {
              std::cerr << "Error in ~GatherElementsObj: " << e.what()
                        << std::endl;
          }
      }
  }
    void initInfiniOp(const Runtime context) override;
    std::string toString() const override;
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};
} // namespace infini
