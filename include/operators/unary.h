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
    ClipObj(GraphObj *graph, Tensor input, Tensor output,
            std::optional<float> min, std::optional<float> max);
    OP_CLONE(ClipObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    std::optional<float> getMin() const { return minValue; };
    std::optional<float> getMax() const { return maxValue; };
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    std::optional<float> minValue, maxValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class HardtanhObj : public OperatorObj {
  public:
    HardtanhObj(GraphObj *graph, Tensor input, Tensor output, float min,
                float max);
    OP_CLONE(HardtanhObj);
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

class FlipObj : public OperatorObj {
  public:
    FlipObj(GraphObj *graph, Tensor input, Tensor output, vector<int> axis);
    OP_CLONE(FlipObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    vector<int> getAxis() const { return axisValue; };
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> axisValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class FillObj : public OperatorObj {
  public:
    FillObj(GraphObj *graph, Tensor input, Tensor output, float value);
    OP_CLONE(FillObj);
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
    OP_CLONE(L2LossObj);
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
    OP_CLONE(TransformObj);
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
        Float2Int64,
        Float2Int32,
        Float2Int16,
        Float2Int8,
        Int322Float,
        Int322Int8,
        Int322Int16,
        Int162Float,
        Int162Int32,
        Int82Float,
        Int82Int16,
        Int82Int32,
        Uint82Float,
        Uint82Int32,
        Uint82Int64,
        Int322Int64,
        Int642Int32,
        Int642Uint32,
        Int642Float,
        Uint322Int64,
    };
    CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type);
    OP_CLONE(CastObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;
    vector<DataType> inferDataType(const TensorVec &inputs) const override;

    std::string toString() const override;
    CastType getType() const { return castType; }
    DataType getOutputDataType() const;
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
    OP_CLONE(CumsumObj);
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

class ArangeObj : public OperatorObj {
  public:
    ArangeObj(GraphObj *graph, float start, float step, int length,
              Tensor output);
    OP_CLONE(ArangeObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 0; }
    int numOutputs() const override { return 1; }
    float getStartValue() { return startValue; }
    float getStepValue() { return stepValue; }
    int getLength() { return lengthValue; }

  private:
    float startValue, stepValue;
    int lengthValue;
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class ShapeObj : public OperatorObj {
  public:
    ShapeObj(GraphObj *graph, Tensor input, Tensor output);
    OP_CLONE(ShapeObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }
};

class PReluObj : public OperatorObj {
  public:
    PReluObj(GraphObj *graph, Tensor input, Tensor alpha, Tensor output);
    OP_CLONE(PReluObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class LogObj : public OperatorObj {
  public:
    enum LogType {
        LogE = 0,
        Log2,
        Log10,
    };
    LogObj(GraphObj *graph, Tensor input, Tensor output, LogType type);
    OP_CLONE(LogObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    LogType getType() const { return logType; }
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    LogType logType;
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
DEFINE_UNARY_OBJ(Neg, OpType::Neg)
DEFINE_UNARY_OBJ(Reciprocal, OpType::Reciprocal)
DEFINE_UNARY_OBJ(Sqrt, OpType::Sqrt)
DEFINE_UNARY_OBJ(Rsqrt, OpType::Rsqrt)
DEFINE_UNARY_OBJ(Round, OpType::Round)
DEFINE_UNARY_OBJ(Square, OpType::Square)
}; // namespace infini
