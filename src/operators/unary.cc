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

ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor output, float min,
                 float max)
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

FlipObj::FlipObj(GraphObj *graph, Tensor input, Tensor output, vector<int> axis)
    : OperatorObj(OpType::Flip, {input}, {output}), axisValue(axis) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FlipObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string FlipObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> FlipObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> FlipObj::getOpAttrVector() const {
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

TransformObj::TransformObj(GraphObj *graph, Tensor input, Tensor output,
                           float alpha, float beta)
    : OperatorObj(OpType::Transform, {input}, {output}), alphaValue(alpha),
      betaValue(beta) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
TransformObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string TransformObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> TransformObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> TransformObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

CastObj::CastObj(GraphObj *graph, Tensor input, Tensor output, CastType type)
    : OperatorObj(OpType::Cast, {input}, {output}), castType(type) {
    IT_ASSERT(checkValid(graph, DataType::Int32));
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

CumsumObj::CumsumObj(GraphObj *graph, Tensor input, Tensor output, int axis,
                     bool exclusive, bool reverse)
    : OperatorObj(OpType::Cumsum, {input}, {output}), axisValue(axis),
      exclusiveValue(exclusive), reverseValue(reverse) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> CumsumObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string CumsumObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> CumsumObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> CumsumObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

// CumprodObj::CumprodObj(GraphObj *graph, Tensor input, Tensor output, int
// axis, bool exclusive, bool reverse)
//     : OperatorObj(OpType::Cumprod, {input}, {output}), axisValue(axis),
//     exclusiveValue(exclusive), reverseValue(reverse)  {
//     IT_ASSERT(checkValid(graph));
// }
//
// optional<vector<Shape>> CumprodObj::inferShape(const TensorVec &inputs) const
// {
//     const auto A = inputs[0];
//     return {{A->getDims()}};
// }
//
// std::string CumprodObj::toString() const {
//     std::ostringstream os;
//     os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
//     os << "(";
//     os << "output=" << outputs[0]->getGuid() << ")";
//     return os.str();
// }
//
// vector<int> CumprodObj::getWorkloadVector() const {
//     vector<int> ret{enum_to_underlying(type)};
//     const Shape shape = outputs[0]->getDims();
//     ret.insert(ret.end(), shape.begin(), shape.end());
//     return ret;
// }
//
// vector<int> CumprodObj::getOpAttrVector() const {
//     return {enum_to_underlying(type)};
// }

ArangeObj::ArangeObj(GraphObj *graph, float start, float step, int length,
                     Tensor output)
    : OperatorObj(OpType::Arange, {}, {output}), startValue(start),
      stepValue(step), lengthValue(length) {
    IT_ASSERT(checkValid(graph, DataType::Float32));
}

optional<vector<Shape>> ArangeObj::inferShape(const TensorVec &inputs) const {
    Shape temp = {lengthValue};
    return {{temp}};
}

std::string ArangeObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(outputs[0]->getDims()) << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ArangeObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> ArangeObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

}; // namespace infini
