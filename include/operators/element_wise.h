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
DEFINE_ELEMENT_WISE_OBJ(Div, OpType::Div)
DEFINE_ELEMENT_WISE_OBJ(Pow, OpType::Pow)
}; // namespace infini
