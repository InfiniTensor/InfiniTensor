#include "code_gen/nnet/Visitor/AsTVMVisitor.h"

namespace nnet {

std::string AsTVMVisitor::visit_(const Constant &c) {
    return std::to_string(c->getValue());
}
std::string AsTVMVisitor::visit_(const BinaryOp &c) {
    switch (c->getOpType()) {
    case OpType::Add:
        return "(" + dispatch(c->getLhs()) + " + " + dispatch(c->getRhs()) +
               ")";
    case OpType::Sub:
        return "(" + dispatch(c->getLhs()) + " - " + dispatch(c->getRhs()) +
               ")";
    case OpType::Mul:
        return "(" + dispatch(c->getLhs()) + " * " + dispatch(c->getRhs()) +
               ")";
    case OpType::Div:
        return "(" + dispatch(c->getLhs()) + " // " + dispatch(c->getRhs()) +
               ")";
    case OpType::Mod:
        return "(" + dispatch(c->getLhs()) + " % " + dispatch(c->getRhs()) +
               ")";
    default:
        assert(false);
    }
}
std::string AsTVMVisitor::visit_(const Func &c) {
    switch (c->getFuncType()) {
    case FuncType::Relu:
        // TODO: Deduce the dtype
        return "te.max(" + dispatch(c->getObject()) +
               ", tvm.tir.const(0, 'float32'))";
    case FuncType::Tanh:
        return "te.tanh(" + dispatch(c->getObject()) + ")";
    default:
        assert(false);
    }
}
std::string AsTVMVisitor::visit_(const RangeOp &c) {
    auto outerStage = curStage;
    curStage = nStage++;

    std::string stmt;
    std::string stageName = "s" + std::to_string(curStage);
    std::vector<std::string> reduceVars;
    for (auto &&[var, range] : c->getSumVarRanges()) {
        std::string varName = stageName + "_" + var->getName();
        stmt += varName + " = " + "te.reduce_axis((" +
                std::to_string(range.first) + ", " +
                std::to_string(range.second) + "), name=\"" + varName + "\")\n";
        reduceVars.emplace_back(varName);
        pythonVars.emplace_back(varName);
    }
    std::vector<int> shape;
    stmt += stageName + " = te.compute((";
    for (size_t i = 0, n = c->getLoopVarRanges().size(); i < n; i++) {
        auto &&[var, range] = c->getLoopVarRanges()[i];
        std::string varName = stageName + "_" + var->getName();
        offset[varName] = -range.first + c->getPaddings(i);
        auto len = range.second - range.first + 2 * c->getPaddings(i);
        stmt += std::to_string(len) + ", ";
        shape.emplace_back(len);
    }
    stmt += "), lambda ";
    bool first = true;
    for (auto &&[var, range] : c->getLoopVarRanges()) {
        std::string varName = stageName + "_" + var->getName();
        stmt += (first ? "" : ", ") + varName;
        first = false;
    }
    std::string summand = dispatch(c->getSummand());
    if (!reduceVars.empty()) {
        summand = "te.sum(" + summand + ", axis=(";
        for (auto &&var : reduceVars) {
            summand += var + ", ";
        }
        summand += "))";
    }
    if (c->hasPaddings()) {
        std::string guard = "tir.if_then_else(tir.all(";
        bool first = true;
        for (size_t i = 0, n = c->getLoopVarRanges().size(); i < n; i++) {
            auto &&[var, range] = c->getLoopVarRanges()[i];
            std::string varName = stageName + "_" + var->getName();
            if (auto pad = c->getPaddings(i); pad > 0) {
                guard += (first ? "" : ", ") + varName +
                         " >= " + std::to_string(range.first) + ", " + varName +
                         " < " + std::to_string(range.second);
                first = false;
            }
        }
        // TODO: Deduce the dtype
        guard += "), " + summand + ", tvm.tir.const(0.0, \"float32\"))";
        summand = guard;
    }
    stmt += ": " + summand + ")";
    stmts += stmt + "\n";

    pythonVars.emplace_back(stageName);
    output = stageName;
    outputShape = std::move(shape);
    curStage = outerStage;
    return stageName;
}
std::string AsTVMVisitor::visit_(const Subscript &c) {
    std::string str = dispatch(c->getObject()) + "[";
    for (size_t i = 0, n = c->getIndex().size(); i < n; i++) {
        const auto &idx = c->getIndex()[i];
        str += (i == 0 ? "" : ", ") + dispatch(idx);
        if (c->getObject()->getType() == NodeType::RangeOpNodeType) {
            auto rangeOp = as<RangeOpNode>(c->getObject());
            str += " - " +
                   std::to_string(rangeOp->getLoopVarRanges()[i].second.first -
                                  rangeOp->getPaddings(i));
        }
    }
    str += "]";
    return str;
}
std::string AsTVMVisitor::visit_(const Var &c) {
    std::string stageName = "s" + std::to_string(curStage);
    std::string varName = stageName + "_" + c->getName();
    if (offset.count(varName)) {
        return "(" + varName + " - " + std::to_string(offset.at(varName)) + ")";
    } else {
        return varName;
    }
}
std::string AsTVMVisitor::visit_(const Tensor &c) {
    pythonVars.emplace_back(c->getName());
    inputs.emplace_back(c->getName());
    inputShapes.emplace_back(c->getShape());
    std::string stmt = c->getName() + " = te.placeholder((";
    for (auto &&dim : c->getShape()) {
        stmt += std::to_string(dim) + ", ";
    }
    stmt += "), name='" + c->getName() + "')";
    stmts += stmt + "\n";
    return c->getName();
}
std::string AsTVMVisitor::getStmts() const {
    std::string ret;

    // Workaround because closure capturing does not work in an `exec`
    // https://stackoverflow.com/questions/2749655/why-are-closures-broken-within-exec
    ret += "global ";
    bool first = true;
    for (auto &&var : pythonVars) {
        ret += (first ? "" : ", ") + var;
        first = false;
    }
    ret += "\n";

    ret += stmts;
    ret += "ret = [" + output;
    for (auto &&input : inputs) {
        ret += ", " + input;
    }
    ret += "]\n";
    return ret;
}

} // namespace nnet