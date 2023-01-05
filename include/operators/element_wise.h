#pragma once
#include "core/operator.h"

namespace infini {
class ElementWiseObj : public OperatorObj {
  public:
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

class AddNObj : public OperatorObj {
  public:
    AddNObj(GraphObj *graph, int tensorNum, Tensor output, ...);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return num; }
    int numOutputs() const override { return 1; }

  private:
    int num;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class MulNObj : public OperatorObj {
  public:
    MulNObj(GraphObj *graph, int tensorNum, Tensor output, ...);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return num; }
    int numOutputs() const override { return 1; }

  private:
    int num;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

#define DEFINE_ELEMENT_WISE_OBJ(prefix, type)                                  \
    class prefix##Obj : public ElementWiseObj {                                \
      public:                                                                  \
        prefix##Obj(GraphObj *graph, Tensor input0, Tensor input1,             \
                    Tensor output)                                             \
            : ElementWiseObj(type, graph, input0, input1, output) {}           \
    };

DEFINE_ELEMENT_WISE_OBJ(Add, OpType::Add)
DEFINE_ELEMENT_WISE_OBJ(Sub, OpType::Sub)
DEFINE_ELEMENT_WISE_OBJ(Mul, OpType::Mul)
DEFINE_ELEMENT_WISE_OBJ(DivDemo, OpType::DivDemo)
DEFINE_ELEMENT_WISE_OBJ(DivNoNan, OpType::DivNoNan)
DEFINE_ELEMENT_WISE_OBJ(Div, OpType::Div)
DEFINE_ELEMENT_WISE_OBJ(Pow, OpType::Pow)
DEFINE_ELEMENT_WISE_OBJ(Maximum, OpType::Maximum)
DEFINE_ELEMENT_WISE_OBJ(Minimum, OpType::Minimum)
DEFINE_ELEMENT_WISE_OBJ(Power, OpType::Power)
DEFINE_ELEMENT_WISE_OBJ(FloorDiv, OpType::FloorDiv)
DEFINE_ELEMENT_WISE_OBJ(FloorDivTrunc, OpType::FloorDivTrunc)
DEFINE_ELEMENT_WISE_OBJ(FloorMod, OpType::FloorMod)
DEFINE_ELEMENT_WISE_OBJ(FloorModTrunc, OpType::FloorModTrunc)
}; // namespace infini
