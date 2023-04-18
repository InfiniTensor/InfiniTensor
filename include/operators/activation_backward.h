#pragma once
#include "core/operator.h"

namespace infini {
class ActivationBackwardObj : public OperatorObj {
  public:
    ActivationBackwardObj(OpType type, GraphObj *graph, Tensor y, Tensor diff_y,
                          Tensor x, Tensor diff_x);
    OP_CLONE(ActivationBackwardObj);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) const override;

    std::string toString() const override;
    int numInputs() const override { return 3; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

#define DEFINE_ACTIVATION_BACKWARD_OBJ(prefix, type)                           \
    class prefix##Obj : public ActivationBackwardObj {                         \
      public:                                                                  \
        prefix##Obj(GraphObj *graph, Tensor y, Tensor diff_y, Tensor x,        \
                    Tensor diff_x)                                             \
            : ActivationBackwardObj(type, graph, y, diff_y, x, diff_x) {}      \
    };

DEFINE_ACTIVATION_BACKWARD_OBJ(ReluBackward, OpType::ReluBackward)
DEFINE_ACTIVATION_BACKWARD_OBJ(SigmoidBackward, OpType::SigmoidBackward)
DEFINE_ACTIVATION_BACKWARD_OBJ(TanhBackward, OpType::TanhBackward)
}; // namespace infini
