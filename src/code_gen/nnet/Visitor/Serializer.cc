#include "code_gen/nnet/Visitor/Serializer.h"
#include "code_gen/nnet/expr.h"
#include "nlohmann/json.hpp"
#include <fstream>

namespace nnet {

int Serializer::id = 0;

Serializer::Serializer(int _verobse)
    : Functor(_verobse), jPtr(std::make_unique<json>()), j(*jPtr) {}

Serializer::~Serializer() = default;

string Serializer::visit_(const Constant &c) {
    string key = std::to_string(id++);
    j[key]["type"] = c->getType();
    j[key]["val"] = c->getValue();
    return key;
}

string Serializer::visit_(const BinaryOp &c) {
    string key = std::to_string(id++);
    j[key]["type"] = c->getType();
    j[key]["opType"] = (int)c->getOpType();
    j[key]["lhs"] = dispatch(c->getLhs());
    j[key]["rhs"] = dispatch(c->getRhs());
    return key;
}

string Serializer::visit_(const RangeOp &c) {
    string key = std::to_string(id++);
    j[key]["type"] = c->getType();
    j[key]["paddings"] = c->getPaddings();
    j[key]["summand"] = dispatch(c->getSummand());
    for (auto var : c->getLoopVarRanges()) {
        j[key]["loopVarRanges"][var.first->getName()] = var.second;
    }
    for (auto var : c->getSumVarRanges()) {
        j[key]["sumVarRanges"][var.first->getName()] = var.second;
    }
    return key;
}

string Serializer::visit_(const Subscript &c) {
    string key = std::to_string(id++);
    j[key]["type"] = c->getType();
    j[key]["subExprsNum"] = c->getDims();
    j[key]["object"] = dispatch(c->getObject());
    vector<string> indexes;
    for (auto index : c->getIndex()) {
        indexes.emplace_back(dispatch(index));
    }
    j[key]["indexes"] = indexes;
    return key;
}

string Serializer::visit_(const Var &c) {
    string key = std::to_string(id++);
    j[key]["type"] = c->getType();
    j[key]["name"] = c->getName();
    return key;
}

string Serializer::visit_(const Tensor &c) {
    const string key = std::to_string(id++);
    j[key]["type"] = c->getType();
    j[key]["name"] = c->getName();
    j[key]["shape"] = c->getShape();
    j[key]["paddings"] = c->getPaddings();
    const auto &routine = c->getSource();
    j[key]["source"] = dispatchRoutine(routine);
    return key;
}

bool Serializer::serialize(const Expr &expr, const string &filePath,
                           const string &msg) {
    // Metadata
    j["Version"] = VERSION;
    j["Msg"] = msg;
    // Expressions and routines
    id = 0;
    dispatch(expr);
    std::ofstream fout(filePath);
    fout << std::setw(4) << j << std::endl;
    return true;
}

string Serializer::dispatchRoutine(const Routine &c) {
    if (!c)
        return "-1";
    const string key = std::to_string(id++);
    j[key]["type"] = c->getType();

    vector<string> inputs;
    for (const auto &tensor : c->getInputs())
        inputs.emplace_back(dispatch(tensor));
    j[key]["inputs"] = inputs;

    if (const auto &expr = c->getExpr())
        j[key]["expr"] = dispatch(expr);
    else
        j[key]["expr"] = "-1";

    switch (c->getType()) {
    case RoutineType::NoneType:
        nnet_unimplemented_halt();
        break;
    case RoutineType::MatmulNodeType: {
        j[key]["args"] = as<MatmulNode>(c)->getArgs();
        break;
    }
    case RoutineType::ConvNodeType:
        j[key]["args"] = as<ConvNode>(c)->getArgs();
        break;
    case RoutineType::G2bmmNodeType:
        j[key]["args"] = as<G2bmmNode>(c)->getArgs();
        break;
    case RoutineType::GbmmNodeType:
        j[key]["args"] = as<GbmmNode>(c)->getArgs();
        break;
    case RoutineType::ElementWiseNodeType: {
        j[key]["outputShape"] = as<ElementWiseNode>(c)->getOutputShape();
        break;
    }
    default:
        nnet_unimplemented_halt();
    }
    return key;
}

Expr Serializer::deserialize(const string &filePath) {
    std::ifstream fin(filePath);
    fin >> j;
    assert(j["Version"] == VERSION);
    return buildExprTree("0");
}

Expr Serializer::buildExprTree(string key) {
    switch (NodeType(j[key]["type"])) {
    case NodeType::ConstantNodeType: {
        return make_ref<ConstantNode>(j[key]["val"]);
    }
    case NodeType::BinaryOpNodeType: {
        auto lhs = buildExprTree(j[key]["lhs"]);
        auto rhs = buildExprTree(j[key]["rhs"]);
        return make_ref<BinaryOpNode>(j[key]["opType"], lhs, rhs);
    }
    case NodeType::RangeOpNodeType: {
        vector<VarRangePair> loopIters, sumIters;
        for (auto &loopIter : j[key]["loopVarRanges"].items()) {
            loopIters.emplace_back(
                pair(make_ref<VarNode>(loopIter.key()),
                     pair(loopIter.value()[0], loopIter.value()[1])));
        }
        for (auto &sumIter : j[key]["sumVarRanges"].items()) {
            sumIters.emplace_back(
                pair(make_ref<VarNode>(sumIter.key()),
                     pair(sumIter.value()[0], sumIter.value()[1])));
        }
        auto summand = buildExprTree(j[key]["summand"]);
        auto paddings = j[key]["paddings"].get<std::vector<int>>();
        auto rangeOp = makeRangeOperator(loopIters, sumIters, summand);
        rangeOp->setPaddings(paddings);
        return rangeOp;
    }
    case NodeType::SubscriptNodeType: {
        auto indexed = buildExprTree(j[key]["object"]);
        VecExpr subExprs;
        for (int i = 0, iEnd = j[key]["subExprsNum"]; i < iEnd; i++) {
            subExprs.emplace_back(buildExprTree(j[key]["indexes"][i]));
        }
        return make_ref<SubscriptNode>(indexed, subExprs);
    }
    case NodeType::VarNodeType: {
        return make_ref<VarNode>(j[key]["name"]);
    }
    case NodeType::TensorNodeType: {
        auto source = buildRoutine(j[key]["source"]);
        return make_ref<TensorNode>(j[key]["name"], j[key]["shape"],
                                    j[key]["paddings"], source);
    }
    default: {
        nnet_unimplemented_halt();
        break;
    }
    }
    return nullptr;
}

Routine Serializer::buildRoutine(string key) {
    if (key == "-1")
        return nullptr;
    Expr expr = nullptr;
    if (j[key]["expr"] != "-1")
        expr = buildExprTree(j[key]["expr"]);
    vector<Tensor> inputs;
    for (const auto &input : j[key]["inputs"])
        inputs.emplace_back(as<TensorNode>(buildExprTree(input)));

    switch (RoutineType(j[key]["type"])) {
    case RoutineType::NoneType:
        nnet_unimplemented_halt();
        break;
    case RoutineType::MatmulNodeType: {
        assert(inputs.size() == 2);
        auto args = j[key]["args"].get<MatmulArgs>();
        auto ctorArgs =
            std::tuple_cat(std::tie(expr, inputs[0], inputs[1]), args);
        return make_ref_from_tuple<MatmulNode>(ctorArgs);
        break;
    }
    case RoutineType::ConvNodeType: {
        assert(inputs.size() == 2);
        auto args = j[key]["args"].get<ConvArgs>();
        auto ctorArgs =
            std::tuple_cat(std::tie(expr, inputs[0], inputs[1]), args);
        return make_ref_from_tuple<ConvNode>(ctorArgs);
        break;
    }
    case RoutineType::G2bmmNodeType: {
        auto args = j[key]["args"].get<G2bmmArgs>();
        auto ctorArgs =
            std::tuple_cat(std::tie(expr, inputs[0], inputs[1]), args);
        return make_ref_from_tuple<G2bmmNode>(ctorArgs);
        break;
    }
    case RoutineType::GbmmNodeType: {
        auto args = j[key]["args"].get<GbmmArgs>();
        auto ctorArgs =
            std::tuple_cat(std::tie(expr, inputs[0], inputs[1]), args);
        return make_ref_from_tuple<GbmmNode>(ctorArgs);
        break;
    }
    case RoutineType::ElementWiseNodeType: {
        return make_ref<ElementWiseNode>(expr, inputs, j[key]["outputShape"]);
        break;
    }
    default:
        nnet_unimplemented_halt();
    }
    return nullptr;
}

} // namespace nnet