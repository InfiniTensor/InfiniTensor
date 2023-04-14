#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief The base class for unary operators.
 *
 */
class UnaryObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new Unary object.
     *
     * @param type Operator type.
     * @param graph The computation graph that this operator belongs to.
     * @param input The input tensor.
     * @param output The output tensor.
     */
    UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

#define DEFINE_UNARY_OBJ(prefix, type)                                         \
    class prefix##Obj : public UnaryObj {                                      \
      public:                                                                  \
        prefix##Obj(GraphObj *graph, Tensor input, Tensor output)              \
            : UnaryObj(type, graph, input, output) {}                          \
        OP_CLONE(prefix##Obj);                                                 \
    };

DEFINE_UNARY_OBJ(Relu, OpType::Relu)
DEFINE_UNARY_OBJ(Sigmoid, OpType::Sigmoid)
DEFINE_UNARY_OBJ(Tanh, OpType::Tanh)
DEFINE_UNARY_OBJ(Softmax, OpType::Softmax)
DEFINE_UNARY_OBJ(Abs, OpType::Abs)
DEFINE_UNARY_OBJ(PRelu, OpType::PRelu)
}; // namespace infini
