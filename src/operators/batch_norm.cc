#include "operators/batch_norm.h"

namespace infini {
BatchNormObj::BatchNormObj(GraphObj *graph, Tensor input, Tensor output,
                           Tensor mean, Tensor var, Tensor scale, Tensor bias,
                           float momentum, float eps, bool trainingMode)
    : OperatorObj(OpType::BatchNorm, {input, mean, var, scale, bias}, {output}),
      momentum(momentum), eps(eps), trainingMode(trainingMode) {
    if (trainingMode)
        IT_TODO_HALT();

    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
BatchNormObj::inferShape(const TensorVec &inputs) const {
    auto input = inputs[0];
    auto mean = inputs[1];
    auto var = inputs[2];
    auto scale = inputs[3];
    auto bias = inputs[4];
    auto c = std::vector<int>{input->getDims()[1]};
    if (mean->getDims() != c || var->getDims() != c || scale->getDims() != c ||
        bias->getDims() != c)
        return {};
    return {{input->getDims()}};
}

vector<DataType> BatchNormObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 5);
    auto index = inputs[1];
    IT_ASSERT(inputs[1]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[2]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[3]->getDType() == DataType::Float32);
    IT_ASSERT(inputs[4]->getDType() == DataType::Float32);
    return {inputs[0]->getDType()};
}

std::string BatchNormObj::toString() const {
    std::ostringstream os;
    os << "BatchNorm[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "momentum=" << momentum << ",";
    os << "eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "mean=" << inputs[1]->getGuid() << ",";
    os << "var=" << inputs[2]->getGuid() << ",";
    os << "scale=" << inputs[3]->getGuid() << ",";
    os << "bias=" << inputs[4]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

// need eps and momentum?
vector<int> BatchNormObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

// need eps and momentum?
vector<int> BatchNormObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

} // namespace infini
