#include "operators/elu.h"
#include "utils/operator_utils.h"

namespace infini {

EluObj::EluObj(GraphObj *graph, Tensor input, Tensor output, float alpha)
    : OperatorObj(OpType::Elu, {input}, {output}), alpha(alpha) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> EluObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

std::string EluObj::toString() const {
    std::ostringstream os;
    os << "Elu[" << getGuid() << "]";
    os << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "alpha=" << alpha << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> EluObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> EluObj::getOpAttrVector() const {
    return {type.underlying(), static_cast<int>(alpha)};
}

} // namespace infini
