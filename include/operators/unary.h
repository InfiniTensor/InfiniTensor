#pragma once
#include "core/operator.h"

namespace infini {
class UnaryObj : public OperatorObj {
  public:
    UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class ClipObj : public OperatorObj {
  public:
    ClipObj(GraphObj *graph, Tensor input, Tensor output, float min, float max);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    float getMin() const { return minValue; };
    float getMax() const { return maxValue; };
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    float minValue,maxValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class FillObj : public OperatorObj {
  public:
    FillObj(GraphObj *graph, Tensor input, Tensor output, float value);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    float getValue() const { return setValue; };
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    float setValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class L2LossObj : public OperatorObj {
  public:
    L2LossObj(GraphObj *graph, Tensor input, Tensor output);
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
    };

DEFINE_UNARY_OBJ(Relu, OpType::Relu)
DEFINE_UNARY_OBJ(Sigmoid, OpType::Sigmoid)
DEFINE_UNARY_OBJ(Tanh, OpType::Tanh)
DEFINE_UNARY_OBJ(Softmax, OpType::Softmax)
DEFINE_UNARY_OBJ(Abs, OpType::Abs)

DEFINE_UNARY_OBJ(Sin, OpType::Sin)
DEFINE_UNARY_OBJ(Cos, OpType::Cos)
DEFINE_UNARY_OBJ(Tan, OpType::Tan)
DEFINE_UNARY_OBJ(ASin, OpType::ASin)
DEFINE_UNARY_OBJ(ACos, OpType::ACos)
DEFINE_UNARY_OBJ(ATan, OpType::ATan)
DEFINE_UNARY_OBJ(SinH, OpType::SinH)
DEFINE_UNARY_OBJ(CosH, OpType::CosH)
DEFINE_UNARY_OBJ(TanH, OpType::TanH)
DEFINE_UNARY_OBJ(ASinH, OpType::ASinH)
DEFINE_UNARY_OBJ(ACosH, OpType::ACosH)
DEFINE_UNARY_OBJ(ATanH, OpType::ATanH)

DEFINE_UNARY_OBJ(Copy, OpType::Copy)
DEFINE_UNARY_OBJ(Ceil, OpType::Ceil)
DEFINE_UNARY_OBJ(Floor, OpType::Floor)
DEFINE_UNARY_OBJ(Erf, OpType::Erf)
DEFINE_UNARY_OBJ(Exp, OpType::Exp)
DEFINE_UNARY_OBJ(Log_e, OpType::Log_e)
DEFINE_UNARY_OBJ(Log_2, OpType::Log_2)
DEFINE_UNARY_OBJ(Log_10, OpType::Log_10)
DEFINE_UNARY_OBJ(Log1p, OpType::Log1p)
DEFINE_UNARY_OBJ(NegTensor, OpType::NegTensor)
}; // namespace infini
