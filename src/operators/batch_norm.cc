#include "operators/batch_norm.h"

namespace infini {
BatchNormObj::BatchNormObj(GraphObj *graph, Tensor input, Tensor output,
                           Tensor mean, Tensor var, Tensor scale, Tensor bias,
                           float momentum, float eps, bool trainingMode)
    : OperatorObj(OpType::BatchNormalization, {input, mean, var, scale, bias},
                  {output}),
      momentum(momentum), eps(eps), trainingMode(trainingMode) {
    if (trainingMode)
        IT_TODO_HALT();

    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> BatchNormObj::inferShape(const TensorVec &inputs) {
    auto input = inputs[0];
    auto mean = inputs[1];
    auto var = inputs[2];
    auto scale = inputs[3];
    auto bias = inputs[4];
    auto c = std::vector<int>{input->getDims()[1]};
    IT_ASSERT(mean->getRank() == 1 && mean->getDims() == c);
    IT_ASSERT(var->getRank() == 1 && var->getDims() == c);
    IT_ASSERT(scale->getRank() == 1 && scale->getDims() == c);
    IT_ASSERT(bias->getRank() == 1 && bias->getDims() == c);
    return {{input->getDims()}};
}

vector<DataType> BatchNormObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 5);
    auto index = inputs[1];
    return {inputs[0]->getDType()};
}

std::string BatchNormObj::toString() const {
    std::ostringstream os;
    os << "batchNormalization[" << getGuid() << "]";
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
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

// need eps and momentum?
vector<int> BatchNormObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
