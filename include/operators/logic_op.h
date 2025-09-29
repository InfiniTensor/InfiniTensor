#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief Base class for logic operators like And, Or, Not, Xor.
 */
class LogicOpObj : public OperatorObj {
  public:
    LogicOpObj(OpType type, GraphObj *graph, Tensor input0, Tensor input1,
               Tensor output);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

#define DEFINE_LOGIC_OP_OBJ(prefix, type)                                     \
    class prefix##Obj : public LogicOpObj {                                  \
      public:                                                                \
        prefix##Obj(GraphObj *graph, Tensor input0, Tensor input1,           \
                    Tensor output)                                           \
            : LogicOpObj(type, graph, input0, input1, output) {}             \
        OP_CLONE(prefix##Obj);                                               \
    };

DEFINE_LOGIC_OP_OBJ(And, OpType::And)
DEFINE_LOGIC_OP_OBJ(Or, OpType::Or)
DEFINE_LOGIC_OP_OBJ(Xor, OpType::Xor)
DEFINE_LOGIC_OP_OBJ(Not, OpType::Not)

}; // namespace infini