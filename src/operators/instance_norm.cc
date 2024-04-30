#include "operators/instance_norm.h"
#include "utils/operator_utils.h"

namespace infini {
InstanceNormObj::InstanceNormObj(GraphObj *graph, Tensor input, Tensor output,  Tensor scale,
                           Tensor bias,
                           float eps)
    : OperatorObj(OpType::InstanceNormalization,
                  TensorVec{input, scale, bias},
                  {output}),
      eps(eps) {
    
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> InstanceNormObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType> InstanceNormObj::inferDataType(const TensorVec &inputs) const {

    return {inputs[0]->getDType()};
}

std::string InstanceNormObj::toString() const {
    std::ostringstream os;
    os << "InstanceNormalization[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "scale=" << inputs[1]->getGuid() << ",";
    os << "bias=" << inputs[2]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

vector<int> InstanceNormObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> InstanceNormObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
