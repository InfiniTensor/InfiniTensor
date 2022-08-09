#pragma once
#include "common.h"
#include "expr.h"
#include <iostream>
#include <sstream>
namespace nnet {

class RoutineNode;
class MatmulNode;
class ElementWiseNode;
using Routine = Ref<RoutineNode>;
using Matmul = Ref<MatmulNode>;
using ElementWise = Ref<ElementWiseNode>;

#define DEFINE_GETTYPE(CLASS)                                                  \
    RoutineType getType() const override { return RoutineType::CLASS##Type; }

class RoutineNode {
  protected:
    Expr expr;
    vector<Tensor> inputs;

  public:
    RoutineNode(Expr _expr, const vector<Tensor> &_inputs);
    virtual string toReadable() const = 0;
    const Expr &getExpr() const { return expr; }
    const vector<Tensor> &getInputs() const { return inputs; }
    virtual RoutineType getType() const = 0;
};

using MatmulArgs = tuple<int,   // b
                         int,   // m
                         int,   // n
                         int,   // k
                         bool,  // transa
                         bool>; // transb

class MatmulNode : public RoutineNode {
    int b, m, n, k;
    bool transa, transb;

  public:
    MatmulNode(Expr _source, Tensor A, Tensor B, int _b, int _m, int _n, int _k,
               bool _transa, bool _transb)
        : RoutineNode(_source, {A, B}), b(_b), m(_m), n(_n), k(_k),
          transa(_transa), transb(_transb) {}
    DEFINE_GETTYPE(MatmulNode);

    string toReadable() const override;

    friend bool operator==(const MatmulNode &lhs, const MatmulNode &rhs);
    MatmulArgs getArgs() { return tuple(b, m, n, k, transa, transb); }
};

using ConvArgs = tuple<int,  // ph
                       int,  // pw
                       int,  // sh
                       int,  // sw
                       int,  // dh
                       int>; // dw

class ConvNode : public RoutineNode {
    int ph, pw;
    int sh, sw;
    int dh, dw;

  public:
    ConvNode(Expr _source, Tensor A, Tensor K, int _ph, int _pw, int _sh = 1,
             int _sw = 1, int _dh = 1, int _dw = 1)
        : RoutineNode(_source, {A, K}), ph(_ph), pw(_pw), sh(_sh), sw(_sw),
          dh(_dh), dw(_dw) {}
    DEFINE_GETTYPE(ConvNode);

    string toReadable() const override;
    vector<int> getShape() const;
    friend bool operator==(const ConvNode &lhs, const ConvNode &rhs);
    ConvArgs getArgs() const;
};

class ElementWiseNode : public RoutineNode {
    vector<int> outputShape;

  public:
    // _outputShape is redundent, but expr is still missing for DLT.
    ElementWiseNode(Expr _source, vector<Tensor> _inputs,
                    vector<int> _outputShape)
        : RoutineNode(_source, _inputs), outputShape(_outputShape) {}
    DEFINE_GETTYPE(ElementWiseNode);

    string toReadable() const override;
    /**
     * @brief Get the Estimated Time of mem bound OP.
     *
     * @return double Time in ms.
     */
    double getEstimatedTime() const;
    const vector<int> &getOutputShape() const { return outputShape; }
};

using G2bmmArgs = tuple<int,  // b
                        int,  // m
                        int,  // w
                        int,  // k
                        int>; // dilation
class G2bmmNode : public RoutineNode {
    int b, m, w, k;

  public:
    G2bmmNode(Expr source, Tensor A, Tensor B, int b, int m, int w, int k,
              int d = 1)
        : RoutineNode(source, {A, B}), b(b), m(m), w(w), k(k) {
        assert(d == 1);
    }
    DEFINE_GETTYPE(G2bmmNode);

    vector<int> getShape() const;
    string toReadable() const override;
    G2bmmArgs getArgs() const;
};

using GbmmArgs = tuple<int,  // b
                       int,  // m
                       int,  // w
                       int,  // n
                       int>; // dilation
class GbmmNode : public RoutineNode {
    int b, m, w, n;

  public:
    GbmmNode(Expr source, Tensor A, Tensor B, int b, int m, int w, int n,
             int d = 1)
        : RoutineNode(source, {A, B}), b(b), m(m), w(w), n(n) {
        assert(d == 1);
    }
    DEFINE_GETTYPE(GbmmNode);

    vector<int> getShape() const;
    string toReadable() const override;
    GbmmArgs getArgs() const;
};

// Pretty output for dbg with shared_ptr
template <typename T, typename std::enable_if_t<
                          std::is_base_of_v<RoutineNode, T>> *_ = nullptr>
std::ostream &operator<<(std::ostream &os, const shared_ptr<T> &a) {
    os << ((!a) ? string("Null shared_ptr") : a->toReadable());
    return os;
}

// Pretty output for dbg with shared_ptr
template <typename T, typename std::enable_if_t<
                          std::is_base_of_v<RoutineNode, T>> *_ = nullptr>
std::ostream &operator<<(std::ostream &os, const Ref<T> &a) {
    os << ((!a) ? string("Null shared_ptr") : a->toReadable());
    return os;
}

} // namespace nnet