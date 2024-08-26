#include "operators/ascend_quant.h"

namespace infini {
AscendQuantObj::AscendQuantObj(GraphObj *graph, Tensor input, Tensor output,
                               const vector<float> &scale,
                               const vector<float> &offset, bool sqrtMode,
                               std::string roundMode)
    : OperatorObj(OpType::AscendQuant, {input}, {output}), scale(scale),
      offset(offset), sqrtMode(sqrtMode), roundMode(roundMode) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> AscendQuantObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType> AscendQuantObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 1);
    return {DataType(3)};
}

std::string AscendQuantObj::toString() const {
    std::ostringstream os;
    os << "AscendQuant"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "scale=" << vecToString(scale) << ",";
    os << "offset=" << vecToString(offset) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> AscendQuantObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), scale.begin(), scale.end());
    ret.insert(ret.end(), offset.begin(), offset.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> AscendQuantObj::getOpAttrVector() const {
    vector<int> ret = {type.underlying()};
    ret.insert(ret.end(), scale.begin(), scale.end());
    ret.insert(ret.end(), offset.begin(), offset.end());
    ret.insert(ret.end(), sqrtMode);
    return ret;
}

} // namespace infini
