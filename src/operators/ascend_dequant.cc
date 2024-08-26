#include "operators/ascend_dequant.h"

namespace infini {
AscendDequantObj::AscendDequantObj(GraphObj *graph, Tensor input, Tensor output,
                                   const vector<float> &scale,
                                   const vector<float> &offset, bool sqrtMode)
    : OperatorObj(OpType::AscendDequant, {input}, {output}), scale(scale),
      offset(offset), sqrtMode(sqrtMode) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> AscendDequantObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType>
AscendDequantObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 1);
    return {DataType(10)};
}

std::string AscendDequantObj::toString() const {
    std::ostringstream os;
    os << "AscendDequant"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "scale=" << vecToString(scale) << ",";
    os << "offset=" << vecToString(offset) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> AscendDequantObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), scale.begin(), scale.end());
    ret.insert(ret.end(), offset.begin(), offset.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> AscendDequantObj::getOpAttrVector() const {
    vector<int> ret = {type.underlying()};
    ret.insert(ret.end(), scale.begin(), scale.end());
    ret.insert(ret.end(), offset.begin(), offset.end());
    ret.insert(ret.end(), sqrtMode);
    return ret;
}

} // namespace infini
