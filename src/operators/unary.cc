#include "operators/unary.h"

namespace infini {
UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(type, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> UnaryObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string UnaryObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> UnaryObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> UnaryObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output,
                 std::optional<float> min, std::optional<float> max)
    : OperatorObj(OpType::Clip, {input}, {output}), minValue(min),
      maxValue(max) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ClipObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string ClipObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ClipObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> ClipObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

HardtanhObj::HardtanhObj(GraphObj *graph, Tensor input, Tensor output,
                         float min, float max)
    : OperatorObj(OpType::Hardtanh, {input}, {output}), minValue(min),
      maxValue(max) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> HardtanhObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string HardtanhObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> HardtanhObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> HardtanhObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

FillObj::FillObj(GraphObj *graph, Tensor input, Tensor output, float value)
    : OperatorObj(OpType::Fill, {input}, {output}), setValue(value) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FillObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string FillObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> FillObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> FillObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

L2LossObj::L2LossObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::L2Loss, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> L2LossObj::inferShape(const TensorVec &inputs) const {
    Shape temp = {1};
    return {{temp}};
}

std::string L2LossObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> L2LossObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> L2LossObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

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

optional<vector<Shape>> CastObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string CastObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> CastObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> CastObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

DataType CastObj::getOutputDataType() const {
    switch (castType) {
    case CastObj::Float2Int64:
        return DataType::Int64;
    case CastObj::Float2Int32:
        return DataType::Int32;
    case CastObj::Float2Int16:
        return DataType::Int16;
    case CastObj::Float2Int8:
        return DataType::Int8;
    case CastObj::Int322Float:
        return DataType::Float32;
    case CastObj::Int322Int8:
        return DataType::Int8;
    case CastObj::Int322Int16:
        return DataType::Int16;
    case CastObj::Int162Float:
        return DataType::Float32;
    case CastObj::Int162Int32:
        return DataType::Int32;
    case CastObj::Int82Float:
        return DataType::Float32;
    case CastObj::Int82Int16:
        return DataType::Int16;
    case CastObj::Int82Int32:
        return DataType::Int32;
    case CastObj::Uint82Float:
        return DataType::Float32;
    case CastObj::Uint82Int32:
        return DataType::Int32;
    case CastObj::Uint82Int64:
        return DataType::Int64;
    case CastObj::Int322Int64:
        return DataType::Int64;
    case CastObj::Int642Int32:
        return DataType::Int32;
    case CastObj::Int642Uint32:
        return DataType::UInt32;
    case CastObj::Int642Float:
        return DataType::Float32;
    case CastObj::Uint322Int64:
        return DataType::Int64;
    default:
        IT_TODO_HALT();
    }
}

ShapeObj::ShapeObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Shape, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ShapeObj::inferShape(const TensorVec &inputs) const {
    return {{{static_cast<int>(inputs[0]->getDims().size())}}};
}

std::string ShapeObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]("
       << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

PReluObj::PReluObj(GraphObj *graph, Tensor input, Tensor alpha, Tensor output)
    : OperatorObj(OpType::PRelu, {input, alpha}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PReluObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string PReluObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PReluObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> PReluObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

LogObj::LogObj(GraphObj *graph, Tensor input, Tensor output, LogType type)
    : OperatorObj(OpType::Log, {input}, {output}), logType(type) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LogObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string LogObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> LogObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> LogObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

}; // namespace infini
