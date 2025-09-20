#pragma once
#include "core/op_type.h"
#include "core/operator.h"

// Definitions for logical and bitwise operator objects (CPU/CUDA wrappers).
// BinaryLogicalObj and UnaryLogicalObj provide common plumbing used by
// specific operator classes such as `AndObj`, `OrObj`, and `NotObj`.
namespace infini {
class BinaryLogicalObj : public OperatorObj {
  public:
    BinaryLogicalObj(OpType type, GraphObj *g, Tensor in0, Tensor in1,
                     Tensor out);

    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    int numInputs() const override { return 2; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

class UnaryLogicalObj : public OperatorObj {
  public:
    UnaryLogicalObj(OpType type, GraphObj *g, Tensor in, Tensor out);
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;
    std::string toString() const override;
    int numInputs() const override { return 1; }
    int numOutputs() const override { return 1; }

  private:
    vector<int> getWorkloadVector() const override;
    vector<int> getOpAttrVector() const override;
};

#define DEFINE_BINARY_LOGICAL(prefix, type)                                    \
    class prefix##Obj : public BinaryLogicalObj {                              \
      public:                                                                  \
        prefix##Obj(GraphObj *g, Tensor in0, Tensor in1, Tensor out)           \
            : BinaryLogicalObj(type, g, in0, in1, out) {}                      \
        OP_CLONE(prefix##Obj);                                                 \
    };

#define DEFINE_UNARY_LOGICAL(prefix, type)                                     \
    class prefix##Obj : public UnaryLogicalObj {                               \
      public:                                                                  \
        prefix##Obj(GraphObj *g, Tensor in, Tensor out)                        \
            : UnaryLogicalObj(type, g, in, out) {}                             \
        OP_CLONE(prefix##Obj);                                                 \
    };

// Binary Operators
DEFINE_BINARY_LOGICAL(And, OpType::And)
DEFINE_BINARY_LOGICAL(Or, OpType::Or)
DEFINE_BINARY_LOGICAL(Xor, OpType::Xor)
DEFINE_BINARY_LOGICAL(BitAnd, OpType::BitwiseAnd)
DEFINE_BINARY_LOGICAL(BitOr, OpType::BitwiseOr)
DEFINE_BINARY_LOGICAL(BitXor, OpType::BitwiseXor)

// Unary Operators
DEFINE_UNARY_LOGICAL(Not, OpType::Not)
DEFINE_UNARY_LOGICAL(BitNot, OpType::BitwiseNot)
}; // namespace infini
