#include "operators/unary.h"

namespace infini {
UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(type, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string UnaryObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> UnaryObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> UnaryObj::getOpAttrVector() const { return {type.underlying()}; }

ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                 std::optional<float> min, std::optional<float> max)
    : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
      maxValue(max) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string ClipObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ClipObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> ClipObj::getOpAttrVector() const { return {type.underlying()}; }

HardtanhObj::HardtanhObj(GraphObj *graph, Tensor input, Tensor output,
                         float min, float max)
    : OperatorObj(OpType::Hardtanh, {input}, {output}), minValue(min),
      maxValue(max) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> HardtanhObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string HardtanhObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> HardtanhObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> HardtanhObj::getOpAttrVector() const { return {type.underlying()}; }

FillObj::FillObj(GraphObj *graph, Tensor input, Tensor output, float value)
    : OperatorObj(OpType::Fill, {input}, {output}), setValue(value) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FillObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string FillObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> FillObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> FillObj::getOpAttrVector() const { return {type.underlying()}; }

L2LossObj::L2LossObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::L2Loss, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> L2LossObj::inferShape(const TensorVec &inputs) {
    Shape temp = {1};
    return {{temp}};
}

std::string L2LossObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> L2LossObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> L2LossObj::getOpAttrVector() const { return {type.underlying()}; }

CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
    : OperatorObj(OpType::Cast, {input}, {output}), castType(type) {
    IT_ASSERT(checkValid(graph));
}

vector<DataType> CastObj::inferDataType(const TensorVec &inputs) const {
    auto input_dataType = inputs[0]->getDType();
    auto output_dataType = getOutputDataType();
    for (const auto &tensor : inputs)
        IT_ASSERT(input_dataType == tensor->getDType());
    return vector(numOutputs(), output_dataType);
}

optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string CastObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> CastObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> CastObj::getOpAttrVector() const { return {type.underlying()}; }

DataType CastObj::getOutputDataType() const {
    switch (castType) {
    case CastType::Float2Float16:
        return DataType::Float16;
    case CastType::Float2Int64:
        return DataType::Int64;
    case CastType::Float2Int32:
        return DataType::Int32;
    case CastType::Float2Int16:
        return DataType::Int16;
    case CastType::Float2Int8:
        return DataType::Int8;
    case CastType::Int322Float:
        return DataType::Float32;
    case CastType::Int322Int8:
        return DataType::Int8;
    case CastType::Int322Int16:
        return DataType::Int16;
    case CastType::Int162Float:
        return DataType::Float32;
    case CastType::Int162Int32:
        return DataType::Int32;
    case CastType::Int82Float:
        return DataType::Float32;
    case CastType::Int82Int16:
        return DataType::Int16;
    case CastType::Int82Int32:
        return DataType::Int32;
    case CastType::Uint82Float:
        return DataType::Float32;
    case CastType::Uint82Int32:
        return DataType::Int32;
    case CastType::Uint82Int64:
        return DataType::Int64;
    case CastType::Int322Int64:
        return DataType::Int64;
    case CastType::Int642Int32:
        return DataType::Int32;
    case CastType::Int642Uint32:
        return DataType::UInt32;
    case CastType::Int642Float:
        return DataType::Float32;
    case CastType::Uint322Int64:
        return DataType::Int64;
    case CastType::Float162Float:
        return DataType::Float32;
    case CastType::BFloat162Float:
        return DataType::Float32;
    case CastType::Float2BFloat16:
        return DataType::BFloat16;
    case CastType::Float2Float:
        return DataType::Float32;
    case CastType::Float322Bool:
        return DataType::Bool;
    default:
        IT_TODO_HALT();
    }
}

ShapeObj::ShapeObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Shape, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ShapeObj::inferShape(const TensorVec &inputs) {
    return {{{static_cast<int>(inputs[0]->getRank())}}};
}

std::string ShapeObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]("
       << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

PReluObj::PReluObj(GraphObj *graph, Tensor input, Tensor alpha, Tensor output)
    : OperatorObj(OpType::PRelu, {input, alpha}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PReluObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string PReluObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PReluObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> PReluObj::getOpAttrVector() const { return {type.underlying()}; }

LeakyReluObj::LeakyReluObj(GraphObj *graph, Tensor input, Tensor output,
                           float alpha)
    : OperatorObj(OpType::LeakyRelu, {input}, {output}), alphaValue(alpha) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LeakyReluObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string LeakyReluObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "alpha=" << alphaValue << ")";
    return os.str();
}

vector<int> LeakyReluObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> LeakyReluObj::getOpAttrVector() const {
    return {type.underlying()};
}

LogObj::LogObj(GraphObj *graph, Tensor input, Tensor output, LogType type)
    : OperatorObj(OpType::Log, {input}, {output}), logType(type) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LogObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string LogObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> LogObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> LogObj::getOpAttrVector() const { return {type.underlying()}; }

EluObj::EluObj(GraphObj *graph, Tensor input, Tensor output, float alpha)
    : OperatorObj(OpType::Elu, {input}, {output}), alpha(alpha) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> EluObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

std::string EluObj::toString() const {
    std::ostringstream os;
    os << "Elu[" << getGuid() << "]";
    os << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "alpha=" << alpha << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> EluObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> EluObj::getOpAttrVector() const {
    return {type.underlying(), static_cast<int>(alpha)};
}

}; // namespace infini
