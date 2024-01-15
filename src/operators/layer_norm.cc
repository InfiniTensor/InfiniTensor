#include "operators/layer_norm.h"
#include "utils/operator_utils.h"

namespace infini {
LayerNormObj::LayerNormObj(GraphObj *graph, Tensor input, Tensor scale,
                           Tensor output, [[maybe_unused]] Tensor bias,
                           float eps, int axis_, int stash_type)
    : OperatorObj(OpType::LayerNormalization,
                  bias ? TensorVec{input, scale, bias}
                       : TensorVec{input, scale},
                  {output}),
      eps(eps), stash_type(stash_type) {
    const auto size = input->getRank();
    axis = get_real_axis(axis_, size);
    IT_ASSERT(
        is_unidirectional_broadcasting(input->getDims(), scale->getDims()));
    if (bias) {
        IT_ASSERT(
            is_unidirectional_broadcasting(input->getDims(), bias->getDims()));
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LayerNormObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

vector<DataType> LayerNormObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2 || inputs.size() == 3);

    return {inputs[0]->getDType()};
}

std::string LayerNormObj::toString() const {
    std::ostringstream os;
    os << "layerNormalization[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "axis=" << axis << ",";
    os << "eps=" << eps << ",";
    os << "stash_type=" << stash_type << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "scale=" << inputs[1]->getGuid() << ",";
    // os << "bias=" << inputs[2]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    return os.str();
}

vector<int> LayerNormObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> LayerNormObj::getOpAttrVector() const {
    return {type.underlying(), axis, stash_type};
}

} // namespace infini
