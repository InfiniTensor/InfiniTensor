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
    float minValue, maxValue;
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

class TransformObj : public OperatorObj {
  public:
    TransformObj(GraphObj *graph, Tensor input, Tensor output, float alpha,
                 float beta);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    float getAlpha() const { return alphaValue; }
    float getBeta() const { return betaValue; }
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    float alphaValue, betaValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class CastObj : public OperatorObj {
  public:
    enum CastType {
        Float2Half = 0,
        Float2HalfIEEE754,
        Float2Double,
        Float2Int64,
        Float2Int32,
        Float2Int16,
        Float2Int8,
        Float2Bool,
        Half2Float,
        Half2Int32,
        Half2Int64,
        Half2Int16,
        Half2Int8,
        Half2Uint8,
        Half2Bool,
        Half2FloatInf,
        Int322Float,
        Int322Half,
        Int322Int8,
        Int322Int16,
        Int162Float,
        Int162Half,
        Int162Int32,
        Int82Float,
        Int82Half,
        Int82Int16,
        Int82Int32,
        Uint82Float,
        Uint82Half,
        Uint82Int32,
        Uint82Int64,
        Bool2Float,
        Bool2Half,
        Bool2Int32,
        Int322Int64,
        Int322Bool,
        Int642Int32,
        Int642Uint32,
        Int642Float,
        Int642Half,
        Uint642Uint32,
        Uint322Int64,
        Uint322Uint64,
        Double2Float
    };
    CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    CastType getType() const { return castType; }
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    CastType castType;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class CumsumObj : public OperatorObj {
  public:
    CumsumObj(GraphObj *graph, Tensor input, Tensor output, int axis,
              bool exclusive, bool reverse);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int getAxis() const { return axisValue; }
    float getExclusive() const { return exclusiveValue; }
    float getReverse() const { return reverseValue; }
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    int axisValue;
    bool exclusiveValue, reverseValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

// class CumprodObj : public OperatorObj {
//   public:
//     CumprodObj(GraphObj *graph, Tensor input, Tensor output, int axis, bool
//     exclusive, bool reverse); optional<vector<Shape>> inferShape(const
//     TensorVec &inputs) const override;
//
//     std::string toString() const override;
//     int getAxis() const { return axisValue; }
//     float getExclusive() const { return exclusiveValue; }
//     float getReverse() const { return reverseValue; }
//     int numInputs() const override { return 1; }
//     int numOutputs() const override { return 1; }
//
//   private:
//     int axisValue;
//     bool exclusiveValue, reverseValue;
//     vector<int> getWorkloadVector() const override;
//     vector<int> getOpAttrVector() const override;
// };

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
// DEFINE_UNARY_OBJ(Softmax, OpType::Softmax)
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
DEFINE_UNARY_OBJ(Reciprocal, OpType::Reciprocal)
DEFINE_UNARY_OBJ(Sqrt, OpType::Sqrt)
DEFINE_UNARY_OBJ(Rsqrt, OpType::Rsqrt)
}; // namespace infini
