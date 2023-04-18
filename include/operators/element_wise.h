#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Base class of **binary** element-wise operators.
 * Unary operators like activations are not the derived classes of
 * ElementWiseObj.
 *
 */
class ElementWiseObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new ElementWise object
     *
     * @param type Operator type.
     * @param graph The computation graph that this operator belongs to.
     * @param input0 The first input tensor.
     * @param input1 The second input tensor.
     * @param output The output tensor.
     */
    ElementWiseObj(OpType type, GraphObj *graph, Tensor input0, Tensor input1,
                   Tensor output);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class MSELossObj : public OperatorObj {
  public:
    enum Reduction { None = 0, Sum, Mean };
    MSELossObj(GraphObj *graph, Tensor input0, Tensor input1,
               Reduction reduction, Tensor output);
    OP_CLONE(MSELossObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    Reduction getReduction() const { return reductionMode; }
    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

  private:
    Reduction reductionMode;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

#define DEFINE_ELEMENT_WISE_OBJ(prefix, type)                                  \
    class prefix##Obj : public ElementWiseObj {                                \
      public:                                                                  \
        prefix##Obj(GraphObj *graph, Tensor input0, Tensor input1,             \
                    Tensor output)                                             \
            : ElementWiseObj(type, graph, input0, input1, output) {}           \
        OP_CLONE(prefix##Obj);                                                 \
    };

DEFINE_ELEMENT_WISE_OBJ(Add, OpType::Add)
DEFINE_ELEMENT_WISE_OBJ(Sub, OpType::Sub)
DEFINE_ELEMENT_WISE_OBJ(Mul, OpType::Mul)
DEFINE_ELEMENT_WISE_OBJ(Div, OpType::Div)
DEFINE_ELEMENT_WISE_OBJ(Pow, OpType::Pow)
DEFINE_ELEMENT_WISE_OBJ(Maximum, OpType::Maximum)
DEFINE_ELEMENT_WISE_OBJ(Minimum, OpType::Minimum)
DEFINE_ELEMENT_WISE_OBJ(Power, OpType::Power)
DEFINE_ELEMENT_WISE_OBJ(FloorDiv, OpType::FloorDiv)
DEFINE_ELEMENT_WISE_OBJ(FloorMod, OpType::FloorMod)
DEFINE_ELEMENT_WISE_OBJ(SquaredDifference, OpType::SquaredDifference)
DEFINE_ELEMENT_WISE_OBJ(Equal, OpType::Equal)
DEFINE_ELEMENT_WISE_OBJ(NotEqual, OpType::NotEqual)
DEFINE_ELEMENT_WISE_OBJ(GreaterThan, OpType::GreaterThan)
DEFINE_ELEMENT_WISE_OBJ(GreaterEqual, OpType::GreaterEqual)
DEFINE_ELEMENT_WISE_OBJ(LessThan, OpType::LessThan)
DEFINE_ELEMENT_WISE_OBJ(LessEqual, OpType::LessEqual)
DEFINE_ELEMENT_WISE_OBJ(And, OpType::And)
DEFINE_ELEMENT_WISE_OBJ(Or, OpType::Or)
DEFINE_ELEMENT_WISE_OBJ(Xor, OpType::Xor)
DEFINE_ELEMENT_WISE_OBJ(Not, OpType::Not)
DEFINE_ELEMENT_WISE_OBJ(BitAnd, OpType::BitAnd)
DEFINE_ELEMENT_WISE_OBJ(BitOr, OpType::BitOr)
DEFINE_ELEMENT_WISE_OBJ(BitXor, OpType::BitXor)
DEFINE_ELEMENT_WISE_OBJ(BitNot, OpType::BitNot)
DEFINE_ELEMENT_WISE_OBJ(BitLeftShift, OpType::BitLeftShift)
DEFINE_ELEMENT_WISE_OBJ(BitRightShift, OpType::BitRightShift)
}; // namespace infini
