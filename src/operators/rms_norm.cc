#include "operators/rms_norm.h"

namespace infini {
RMSNormObj::RMSNormObj(GraphObj *graph, Tensor input, Tensor weight,
                       Tensor output)
    : OperatorObj(OpType::RMSNorm, {input, weight}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RMSNormObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    return {{output_dim}};
}

std::string RMSNormObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> RMSNormObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> RMSNormObj::getOpAttrVector() const { return {type.underlying()}; }

}; // namespace infini
