#include "code_gen/nnet/expr.h"
#include "code_gen/nnet/Visitor/GetTensorsVisitor.h"

namespace nnet {

string serializeVec(vector<Expr> v) {
    if (v.empty())
        return "[]";
    return "[" +
           std::accumulate(v.begin() + 1, v.end(), v[0]->toReadable(),
                           [](const string &a, Expr b) {
                               return a + ',' + b->toReadable();
                           }) +
           "]";
}

string serializeVec(vector<Var> v) {
    VecExpr vv;
    for (const auto &a : v)
        vv.emplace_back(a);
    return serializeVec(vv);
}

std::ostream &operator<<(std::ostream &ios, const ExprNode &expr) {
    ios << expr.toReadable();
    return ios;
}

TensorNode::TensorNode(string _name, vector<int> _shape, vector<int> _paddings,
                       Routine _source)
    : name(_name), shape(_shape), paddings(_paddings), source(_source) {
    if (source && source->getExpr()) {
        if (auto range = as<RangeOpNode>(source->getExpr()))
            for (auto [iter, lr] : range->getLoopVarRanges())
                nnet_assert(lr.first == 0 && lr.second > 0,
                            "Tensor dims should start from 0.");
    }
    if (paddings.size() == 0)
        paddings = vector<int>(shape.size(), 0);
    assert(paddings.size() == shape.size());
}

string TensorNode::toOutputShape() const {
    return "shape=" + serializeVec(shape) + " pad=" + serializeVec(paddings);
}

string TensorNode::toReadable() const {
    string ret = name;
    string property = "<pad=";
    bool hasPaddings = false;
    for (size_t i = 0; i < paddings.size(); ++i) {
        if (i > 0)
            property += ",";
        property += to_string(paddings[i]);
        if (paddings[i])
            hasPaddings = true;
    }
    property += ">";
    return (hasPaddings) ? ret + property : ret;
}

int TensorNode::getData(const Ref<vector<int>> &data, const vector<int> &idx) {
    assert(idx.size() == shape.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] < 0 || idx[i] >= shape[i]) {
            assert(0 - paddings[i] <= idx[i]);
            assert(idx[i] < shape[i] + paddings[i]);
            return 0;
        }
    }
    return data->at(getOffset(idx));
}

size_t TensorNode::getOffset(const vector<int> &idx) {
    auto nDim = idx.size();
    assert(shape.size() == nDim);
    if (idx.empty()) {
        return 0;
    }
    for (size_t i = 0; i < nDim; i++) {
        if (idx[i] < 0 || shape[i] <= idx[i]) {
            return (size_t)-1;
        }
    }
    size_t offset = idx[0];
    size_t dim = 0;
    while (++dim < nDim) {
        offset = offset * shape[dim] + idx[dim];
    }
    return offset;
}

string RangeOpNode::toReadable() const {
    string ret;
    for (int i = 0; i < IterationType::NumIterationType; ++i) {
        ret += (i == Loop) ? "L" : "Sum";
        for (const auto &kv : vars[i]) {
            ret += "<" + kv.first->getName() + ":" +
                   std::to_string(kv.second.first) + ":" +
                   std::to_string(kv.second.second) + ">";
        }
        if (i == Loop && hasPaddings()) {
            ret += "<pad=";
            for (const auto &i : paddings) {
                ret += to_string(i) + ",";
            }
            ret += ">";
        }
    }
    if (auto sub = as<SubscriptNode>(getSummand()); sub) {
        ret += "  ...  " + serializeVec(sub->getIndex()) + "\n    {" +
               sub->getObject()->toReadable() + "}";
    } else {
        ret += "\n    {" + subExprs[Summand]->toReadable() + "}";
    }
    return ret;
};
int RangeOpNode::getNumOutputDims() const { return vars[Loop].size(); }
bool RangeOpNode::hasVar(int index, Var name) const {
    for (const auto &kv : vars[index])
        if (kv.first->equal(name))
            return true;
    return false;
}
int RangeOpNode::getVarIndex(int type, string name) {
    for (size_t i = 0; i < vars[type].size(); ++i)
        if (vars[type][i].first->equal(name))
            return i;
    assert(0);
    return 0;
}
Range RangeOpNode::getRange(const Var &var) const {
    for (const auto &varRanges : vars) {
        for (const auto &varRange : varRanges) {
            if (varRange.first->equal(var))
                return varRange.second;
        }
    }
    nnet_assert(0, "Var is not a iterator.");
    return Range();
}
VarRangePair RangeOpNode::getVarRange(const Var &var) const {
    for (const auto &varRanges : vars) {
        for (const auto &varRange : varRanges) {
            if (varRange.first->equal(var))
                return varRange;
        }
    }
    nnet_assert(0, "Var is not a iterator.");
    return VarRangePair();
}

void SubscriptNode::setObject(Expr e) {
    nnet_assert(as<TensorNode>(e) || as<RangeOpNode>(e),
                "Illegal subscripted object");
    indexed = e;
}
bool SubscriptNode::isRangeOpSubscripted() const {
    return as<RangeOpNode>(indexed) != nullptr;
}
vector<Range> SubscriptNode::getObjectRanges() const {
    vector<Range> ret;
    if (isRangeOpSubscripted()) {
        ret = as<RangeOpNode>(indexed)->getOutputRanges();
    } else {
        auto tensor = as<TensorNode>(indexed);
        for (const auto &len : tensor->getShape())
            ret.emplace_back(0, len);
        for (int i = 0; i < tensor->getDims(); ++i) {
            if (int pad = tensor->getPadding(i)) {
                ret[i].first -= pad;
                ret[i].second += pad;
            }
        }
    }
    return ret;
}

optional<pair<Iterator, int>> BinaryOpNode::getModDivParameter() const {
    auto lhs = as<VarNode>(getLhs());
    auto rhs = as<ConstantNode>(getRhs());
    if (lhs == nullptr) {
        return {};
    }
    if (lhs->getType() != NodeType::VarNodeType) {
        nnet_unimplemented_halt();
    }
    if (rhs->getType() != NodeType::ConstantNodeType) {
        nnet_unimplemented_halt();
    }
    assert(rhs != nullptr);
    return pair(lhs, rhs->getValue());
}

pair<Expr, int> BinaryOpNode::getModDivExpr() const {
    auto constant = as<ConstantNode>(getRhs());
    assert(constant != nullptr);
    return pair(getLhs(), constant->getValue());
}

string BinaryOpNode::toReadable() const {
    string ret = "(";
    ret += subExprs[LHS]->toReadable();
    ret += " ";
    ret += opSymbols[static_cast<std::underlying_type_t<OpType>>(opType)];
    ret += " ";
    ret += subExprs[RHS]->toReadable();
    return ret + ")";
};

bool BinaryOpNode::isSwapable() const {
    switch (getOpType()) {
    case OpType::Add:
    case OpType::Mul:
        return true;
    case OpType::Sub:
    case OpType::Div:
    case OpType::Mod:
        return false;
    default:
        nnet_unimplemented_halt();
        return false;
    }
}

string SubscriptNode::toReadable() const {
    string ret;
    ret += "{";
    ret += indexed->toReadable();
    ret += "}[";
    for (size_t i = 0; i < subExprs.size(); ++i) {
        ret += subExprs[i]->toReadable();
        if (i != subExprs.size() - 1)
            ret += ", ";
        else
            ret += "]";
    }
    return ret;
};

string FuncNode::toReadable() const {
    string ret;
    if (funcType == FuncType::Relu)
        ret += "Relu";
    else if (funcType == FuncType::Tanh)
        ret += "Tanh";
    else
        nnet_unimplemented_halt();
    ret += "(  ...  " + serializeVec(object->getIndex()) + ")\n    {" +
           object->getObject()->toReadable() + "}";
    return ret;
}

Expr operator+(const Expr &lhs, const Expr &rhs) {
    if (lhs == nullptr && rhs == nullptr)
        return nullptr;
    else if (lhs == nullptr)
        return rhs;
    else if (rhs == nullptr)
        return lhs;
    else
        return make_ref<BinaryOpNode>(OpType::Add, lhs, rhs);
}

BinaryOp operator-(const Expr &lhs, const Expr &rhs) {
    return make_ref<BinaryOpNode>(OpType::Sub, lhs, rhs);
}

BinaryOp operator*(const Expr &lhs, const Expr &rhs) {
    return make_ref<BinaryOpNode>(OpType::Mul, lhs, rhs);
}

BinaryOp operator/(const Expr &lhs, const Expr &rhs) {
    return make_ref<BinaryOpNode>(OpType::Div, lhs, rhs);
}

BinaryOp operator%(const Expr &lhs, const Expr &rhs) {
    return make_ref<BinaryOpNode>(OpType::Mod, lhs, rhs);
}

Expr operator+(const Expr &lhs, const int &rhs) {
    if (lhs != nullptr && rhs != 0)
        return make_ref<BinaryOpNode>(OpType::Add, lhs,
                                      make_ref<ConstantNode>(rhs));
    else if (lhs == nullptr)
        return make_ref<ConstantNode>(rhs);
    else
        return lhs;
}

Expr operator+(const int &lhs, const Expr &rhs) { return rhs + lhs; }

Expr operator-(const Expr &lhs, const int &rhs) { return lhs + (-rhs); }

Expr operator-(const int &lhs, const Expr &rhs) {
    if (rhs != nullptr)
        return make_ref<BinaryOpNode>(OpType::Sub, make_ref<ConstantNode>(lhs),
                                      rhs);
    else
        return make_ref<ConstantNode>(lhs);
}

Expr operator*(const Expr &lhs, const int &rhs) {
    if (rhs == 1)
        return lhs;
    else
        return make_ref<BinaryOpNode>(OpType::Mul, lhs,
                                      make_ref<ConstantNode>(rhs));
}

Expr operator*(const int &lhs, const Expr &rhs) {
    if (lhs == 1)
        return rhs;
    else
        return make_ref<BinaryOpNode>(OpType::Mul, make_ref<ConstantNode>(lhs),
                                      rhs);
}

bool operator==(const Var &lhs, const string &rhs) {
    return lhs->getName() == rhs;
}

bool operator==(const string &lhs, const Var &rhs) { return rhs == lhs; }
Expr operator%(const Expr &lhs, const int rhs) {
    return make_ref<BinaryOpNode>(OpType::Mod, lhs,
                                  make_ref<ConstantNode>(rhs));
}
Expr operator/(const Expr &lhs, const int rhs) {
    if (rhs == 1)
        return lhs;
    else
        return make_ref<BinaryOpNode>(OpType::Div, lhs,
                                      make_ref<ConstantNode>(rhs));
}

// Wrappers for type deduction
Subscript makeSubscript(const Expr &tensor, const VecExpr &subscripts) {
    return make_ref<SubscriptNode>(tensor, subscripts);
}

RangeOp makeRangeOperator(const vector<VarRangePair> &_loopIters,
                          const vector<VarRangePair> &_sumIters, Expr _summand,
                          const vector<int> &paddings) {
    return make_ref<RangeOpNode>(_loopIters, _sumIters, _summand, paddings);
}

// Wrappers for type deduction
Tensor makeTensor(const string &name, const vector<int> &shape,
                  const vector<int> &paddings, const Routine &source) {
    if (paddings.size() == 0)
        return make_ref<TensorNode>(name, shape,
                                    vector<int>((int)shape.size(), 0), source);
    else
        return make_ref<TensorNode>(name, shape, paddings, source);
}

int64_t TensorNode::getSize() const {
    int64_t size = 1;
    for (auto len : shape)
        size *= len;
    return size;
}
int RangeOpNode::getPaddings(int dim) const {
    return dim < (int)paddings.size() ? paddings[dim] : 0;
}

vector<int> RangeOpNode::getPaddings() const {
    if (paddings.size() > 0)
        return paddings;
    else
        return vector<int>(getNumOutputDims(), 0);
}

void RangeOpNode::setPaddings(vector<int> _paddings) { paddings = _paddings; }

bool RangeOpNode::hasPaddings() const {
    for (const auto &p : paddings)
        if (p > 0)
            return true;
    return false;
}

int64_t RangeOpNode::getFlops() const {
    int64_t sumCnt = getOutputSize();
    if (vars[Sum].empty())
        sumCnt = 0;
    else
        for (const auto &[var, range] : getSumVarRanges())
            sumCnt *= range.second - range.first;
    return sumCnt;
}

int64_t RangeOpNode::getInputSize(const RangeOp &self) const {
    int64_t ret = 0;
    auto tensors = GetTensorsVisitor().get(self);
    for (const auto &[name, tensor] : tensors)
        ret += tensor->getSize();
    return ret;
}

int64_t RangeOpNode::getOutputSize() const {
    int64_t loopCnt = 1;
    for (const auto &[var, range] : getLoopVarRanges())
        loopCnt *= range.second - range.first;
    return loopCnt;
}

vector<int> RangeOpNode::getOutputShape() const {
    vector<int> ret;
    for (const auto &[var, range] : getLoopVarRanges())
        ret.emplace_back(range.second - range.first);
    return ret;
}

vector<Range> RangeOpNode::getOutputRanges() const {
    vector<Range> ret;
    for (const auto &[var, range] : getLoopVarRanges())
        ret.emplace_back(range);
    for (size_t i = 0; i < paddings.size(); ++i)
        if (paddings[i] > 0) {
            ret[i].first -= paddings[i];
            ret[i].second += paddings[i];
        }
    return ret;
}

void FuncNode::setObject(Expr e) {
    object = as<SubscriptNode>(e);
    nnet_assert(object, "Illegal subscripted object");
}

} // namespace nnet
