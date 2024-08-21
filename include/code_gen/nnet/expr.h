#pragma once
#include "common.h"
#include "ref.h"
#include <iostream>
#include <numeric>
#include <type_traits>

namespace nnet {

class ExprNode;
class VarNode;
class TensorNode;
class OperatorNode;
class RangeOpNode;
class SubscriptNode;
class BinaryOpNode;
class ConstantNode;
class FuncNode;
using Expr = Ref<ExprNode>;
using Var = Ref<VarNode>;
using Tensor = Ref<TensorNode>;
using Operator = Ref<OperatorNode>;
using RangeOp = Ref<RangeOpNode>;
using Subscript = Ref<SubscriptNode>;
using BinaryOp = Ref<BinaryOpNode>;
using Constant = Ref<ConstantNode>;
using Func = Ref<FuncNode>;

class RoutineNode;
using Routine = Ref<RoutineNode>;
enum class RoutineType {
    NoneType = 100,
    MatmulNodeType,
    ConvNodeType,
    G2bmmNodeType,
    GbmmNodeType,
    ElementWiseNodeType // unmatchable
};
constexpr inline int MatchableRoutineTypeCnt = 4;
constexpr inline int RoutineTypeCnt = MatchableRoutineTypeCnt + 1;
inline RoutineType idToRoutineType(int i) {
    return static_cast<RoutineType>(i + 1 +
                                    static_cast<int>(RoutineType::NoneType));
}
inline int routineTypeToId(const RoutineType &routineType) {
    return static_cast<int>(routineType) -
           static_cast<int>(RoutineType::NoneType) - 1;
}

using VecExpr = vector<Expr>;

// common data structure
using Iterator = Var; // RE: remove this alias
template <typename T, typename U> using PtrMap = std::map<T, U, ptr_less<T>>;
template <typename T, typename U>
// When keys are pointers, compare keys according to its value instead of
// address Specially, the name of Var are compared due to the overload of op=
// and hash.
using PtrUmap = std::unordered_map<T, U, ptr_hash<T>, ptr_equal<T>>;
template <typename T>
using PtrUset = std::unordered_set<T, ptr_hash<T>, ptr_equal<T>>;
using Appearance = PtrMap<Var, vector<pair<Tensor, int>>>;
using StrideTable =
    PtrMap<Var, vector<tuple<TensorNode *, int, int>>>; // Tensor, dim, stride

// AST node opeartor
bool operator==(const Var &lhs, const string &rhs);
bool operator==(const string &lhs, const Var &rhs);
Expr operator+(const Expr &lhs, const Expr &rhs);
BinaryOp operator-(const Expr &lhs, const Expr &rhs);
BinaryOp operator*(const Expr &lhs, const Expr &rhs);
BinaryOp operator/(const Expr &lhs, const Expr &rhs);
BinaryOp operator%(const Expr &lhs, const Expr &rhs);

Expr operator+(const Expr &lhs, const int &rhs);
Expr operator+(const int &lhs, const Expr &rhs);
Expr operator-(const Expr &lhs, const int &rhs);
Expr operator-(const int &lhs, const Expr &rhs);
Expr operator*(const Expr &lhs, const int &rhs);
Expr operator*(const int &lhs, const Expr &rhs);
Expr operator%(const Expr &lhs, const int rhs);
Expr operator/(const Expr &lhs, const int rhs);

string serializeVec(vector<Expr> v);
string serializeVec(vector<Var> v);
template <typename T> inline string serializeVec(vector<T> v) {
    if (v.empty())
        return "[]";
    return "[" +
           std::accumulate(
               v.begin() + 1, v.end(), to_string(v[0]),
               [](const string &a, int b) { return a + ',' + to_string(b); }) +
           "]";
}

// For RTTI and visitor pattern
enum class NodeType {
    ConstantNodeType,
    BinaryOpNodeType,
    RangeOpNodeType,
    SubscriptNodeType,
    TensorNodeType,
    VarNodeType,
    FuncNodeType
};

enum class FuncType { Relu, Tanh };

#define DEFINE_GETTYPE(CLASS)                                                  \
    NodeType getType() const override { return NodeType::CLASS##Type; }

class ExprNode {
  public:
    virtual ~ExprNode() {}
    ExprNode &operator=(const ExprNode &rhs) = delete;

    virtual HashType hash() const = 0; // RE: remove?
    virtual string toReadable() const = 0;
    friend std::ostream &operator<<(std::ostream &ios, const ExprNode &expr);

    virtual NodeType getType() const = 0;
};

class VarNode : public ExprNode {
    std::string name;

  public:
    VarNode(std::string _name) : name(_name){};
    virtual ~VarNode() {}
    DEFINE_GETTYPE(VarNode);

    const std::string &getName() const { return name; }
    HashType hash() const override { return genhash(name); };
    string toReadable() const override { return name; };
    bool equal(const Var &rhs) const { return name == rhs->getName(); }
    bool neq(const Var &rhs) const { return !equal(rhs); }
    bool less(const Var &rhs) const { return name < rhs->getName(); }
    bool equal(const string &rhs) const { return name == rhs; }
    bool operator==(const VarNode &rhs) const { return name == rhs.getName(); }
    bool operator<(const VarNode &rhs) const { return name < rhs.getName(); }
};

enum class TensorType { Input, Weight, Intermediate };

class TensorNode : public ExprNode {
    string name;
    vector<int> shape, paddings;
    TensorType type;
    Routine source; // if NO source, then this is a input/weight tensor

  public:
    TensorNode(string _name, vector<int> _shape, vector<int> _paddings = {},
               Routine _source = nullptr);
    virtual ~TensorNode() {}
    DEFINE_GETTYPE(TensorNode);

    bool operator==(const string &rhs) { return name == rhs; }
    friend bool operator==(const string &lhs, const TensorNode &rhs) {
        return lhs == rhs.name;
    }

    HashType hash() const override { return genhash(name); }
    string toReadable() const override;
    string toOutputShape() const;
    const std::string &getName() const { return name; }
    std::vector<int> &getPadding() { return paddings; }
    int getPadding(int i) const { return paddings[i]; }
    const vector<int> &getPaddings() const { return paddings; }
    void setPadding(int i, int p) { paddings[i] = p; }
    const vector<int> &getShape() const { return shape; }
    int getShape(int i) const { return shape[i]; }
    int64_t getSize() const;
    int getDims() const { return shape.size(); }
    const Routine &getSource() const { return source; }
    int getData(const Ref<vector<int>> &data, const vector<int> &idx);
    size_t getOffset(const vector<int> &idx);
};

enum class OpType { Range, Add, Mul, Div, Mod, Sub };
const char opSymbols[] = "#+*/%-";

class OperatorNode : public ExprNode {
  protected:
    const OpType opType;
    VecExpr subExprs;

  public:
    OperatorNode(OpType _opType) : opType(_opType){};
    OperatorNode(OpType _opType, VecExpr _subExprs)
        : opType(_opType), subExprs(_subExprs){};

    int getSubExprsNum() { return subExprs.size(); };
    const VecExpr &getSubExprs() { return subExprs; }
    const Expr &getSubExprs(int i) const { return subExprs[i]; }
    OpType getOpType() const { return opType; };
    void setOperands(int i, Expr e) { subExprs[i] = e; }
};

using Range = pair<int, int>;
using VarRangePair = pair<Var, Range>;
inline int getLength(const Range &range) { return range.second - range.first; }
struct IterationType {
    enum { Loop, Sum };
    constexpr static int NumIterationType = 2;
};
class RangeOpNode : public OperatorNode {
  public:
    enum { Summand, END_POS };
    constexpr static int Loop = IterationType::Loop;
    constexpr static int Sum = IterationType::Sum;

  private:
    vector<VarRangePair> vars[IterationType::NumIterationType];
    vector<int> paddings;

  public:
    RangeOpNode(Expr _summand) : OperatorNode(OpType::Range, {_summand}){};
    RangeOpNode(const vector<VarRangePair> &_loopIters,
                const vector<VarRangePair> &_sumIters, Expr _summand,
                const vector<int> &paddings)
        : OperatorNode(OpType::Range, {_summand}), vars{_loopIters, _sumIters},
          paddings(paddings){};
    DEFINE_GETTYPE(RangeOpNode);

    virtual HashType hash() const override {
        nnet_unimplemented_halt();
        return 0;
    };
    string toReadable() const override;
    const Expr &getSummand() const { return subExprs[Summand]; }
    const vector<VarRangePair> &getVarRanges(int _index) const {
        return vars[_index];
    }
    const vector<VarRangePair> &getLoopVarRanges() const {
        return vars[IterationType::Loop];
    }
    const vector<VarRangePair> &getSumVarRanges() const {
        return vars[IterationType::Sum];
    }
    int getNumOutputDims() const;
    bool hasVar(int index, Var name) const;
    bool hasLoopVar(Var name) const { return hasVar(Loop, name); }
    bool hasSumVar(Var name) const { return hasVar(Sum, name); }
    bool hasLoopVar(string name) const {
        return hasVar(Loop, make_ref<VarNode>(name));
    }
    bool hasSumVar(string name) const {
        return hasVar(Sum, make_ref<VarNode>(name));
    }
    int getVarIndex(int type, string name);
    void setSummand(Expr e) { subExprs[Summand] = e; }
    void setLoopIterator(const vector<VarRangePair> &vecExpr) {
        vars[Loop] = vecExpr;
    }
    void setSumIterator(const vector<VarRangePair> &vecExpr) {
        vars[Sum] = vecExpr;
    }
    void setIterator(const vector<VarRangePair> &loop,
                     const vector<VarRangePair> &sum) {
        setLoopIterator(loop);
        setSumIterator(sum);
    }

    const VarRangePair &getVarRange(int _index, int i) const {
        return vars[_index][i];
    }
    const Var &getLoopVar(int i) const { return vars[Loop][i].first; }
    Range getRange(const Var &var) const;
    VarRangePair getVarRange(const Var &var) const;
    bool hasPaddings() const;
    int getPaddings(int dim) const;
    vector<int> getPaddings() const;
    void setPaddings(vector<int> _paddings);
    void setVarRange(int _index, int i, VarRangePair pair) {
        vars[_index][i] = pair;
    }
    int64_t getFlops() const;
    int64_t getInputSize(const RangeOp &self) const;
    int64_t getOutputSize() const;
    vector<int> getOutputShape() const;
    // Including paddings
    vector<Range> getOutputRanges() const;
};

class BinaryOpNode : public OperatorNode {
    enum { LHS, RHS, END_POS };

  public:
    BinaryOpNode(OpType _opType, Expr _lhs, Expr _rhs)
        : OperatorNode(_opType, {_lhs, _rhs}){};
    virtual ~BinaryOpNode() {}
    DEFINE_GETTYPE(BinaryOpNode);

    virtual HashType hash() const override {
        return genhash((HashType)opType,
                       genhash(subExprs[LHS]->hash(), subExprs[RHS]->hash()));
    };
    virtual string toReadable() const override;
    const Expr &getLhs() const { return getSubExprs(LHS); };
    const Expr &getRhs() const { return getSubExprs(RHS); };
    void setLhs(Expr e) { setOperands(LHS, e); };
    void setRhs(Expr e) { setOperands(RHS, e); };
    // If Var/constant, use this one
    optional<pair<Var, int>> getModDivParameter() const;
    // If (Var+constant)/constant, use this one
    pair<Expr, int> getModDivExpr() const;
    bool isSwapable() const;
};

class ConstantNode : public ExprNode {
    int val;

  public:
    ConstantNode(int _val) : val(_val){};
    ConstantNode(const ConstantNode &rhs) : ExprNode(rhs), val(rhs.val){};
    virtual ~ConstantNode() {}
    DEFINE_GETTYPE(ConstantNode);

    int getValue() const { return val; }
    virtual HashType hash() const override { return genhash(val, 6214587); };
    virtual string toReadable() const override {
        string ret;
        ret += std::to_string(val);
        return ret;
    };
};

class SubscriptNode : public ExprNode {
  protected:
    Expr indexed;
    VecExpr subExprs;

  public:
    SubscriptNode(Expr _indexed, vector<Expr> _subExprs) : subExprs(_subExprs) {
        setObject(_indexed);
    };
    DEFINE_GETTYPE(SubscriptNode);

    virtual HashType hash() const override {
        nnet_unimplemented_continue();
        return -1;
    };
    virtual string toReadable() const override;

    size_t getDims() const { return subExprs.size(); }
    const VecExpr &getIndex() const { return subExprs; }
    const Expr &getIndex(size_t i) const { return subExprs[i]; }
    void setIndex(size_t i, Expr e) { subExprs[i] = e; }
    Expr *getObjectPtr() { return &indexed; }
    Expr getObject() const { return indexed; }
    void setObject(Expr e);
    bool isRangeOpSubscripted() const;
    bool isTensorSubscripted() const { return !isRangeOpSubscripted(); }
    // Get the ranges of objects including paddings
    vector<Range> getObjectRanges() const;
};

class FuncNode : public ExprNode {
  protected:
    Subscript object;
    FuncType funcType;

  public:
    FuncNode(Expr object, FuncType funcType) : funcType(funcType) {
        setObject(object);
    }
    DEFINE_GETTYPE(FuncNode);

    virtual HashType hash() const override {
        nnet_unimplemented_continue();
        return -1;
    };
    virtual string toReadable() const override;

    const Subscript &getObject() const { return object; }
    void setObject(Expr e);

    FuncType getFuncType() const { return funcType; }
};

// Wrappers for type deduction
Subscript makeSubscript(const Expr &tensor, const VecExpr &subscripts);
RangeOp makeRangeOperator(const vector<VarRangePair> &_loopIters,
                          const vector<VarRangePair> &_sumIters, Expr _summand,
                          const vector<int> &paddings = {});
Tensor makeTensor(const string &name, const vector<int> &shape,
                  const vector<int> &paddings = {},
                  const Routine &source = nullptr);

// Pretty output for dbg with shared_ptr
template <typename T, typename std::enable_if_t<std::is_base_of_v<ExprNode, T>>
                          *_ = nullptr>
std::ostream &operator<<(std::ostream &os, const shared_ptr<T> &a) {
    os << ((!a) ? string("nullptr") : a->toReadable());
    return os;
}

// Pretty output for dbg with shared_ptr
template <typename T, typename std::enable_if_t<std::is_base_of_v<ExprNode, T>>
                          *_ = nullptr>
std::ostream &operator<<(std::ostream &os, const Ref<T> &a) {
    os << ((!a) ? string("nullptr") : a->toReadable());
    return os;
}
#undef DEFINE_GETTYPE

} // namespace nnet

namespace std {
template <> struct hash<nnet::VarNode &> {
    size_t operator()(const nnet::VarNode &t) const {
        return std::hash<string>()(t.getName());
    }
};
} // namespace std