#include "operators/dequantize_linear.h"
#include "utils/operator_utils.h"

namespace infini {
DequantizeLinearObj::DequantizeLinearObj(GraphObj *graph, Tensor input,
                                         Tensor scale, Tensor zero_point,
                                         Tensor output, int axis)
    : OperatorObj(OpType::DequantizeLinear,
                  zero_point ? TensorVec{input, scale, zero_point}
                             : TensorVec{input, scale},
                  {output}),
      axis(axis) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
DequantizeLinearObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType>
DequantizeLinearObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2 || inputs.size() == 3);
    return {inputs[1]->getDType()};
}

std::string DequantizeLinearObj::toString() const {
    std::ostringstream os;
    os << "DequantizeLinear[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "scale=" << inputs[1]->getGuid() << ",";
    os << "axis=" << axis << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

vector<int> DequantizeLinearObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> DequantizeLinearObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
