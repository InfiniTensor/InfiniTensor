#include "operators/activation_backward.h"

namespace infini {
ActivationBackwardObj::ActivationBackwardObj(OpType type, GraphObj *graph,
                                             Tensor y, Tensor diff_y, Tensor x,
                                             Tensor diff_x)
    : OperatorObj(type, {y, diff_y, x}, {diff_x}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ActivationBackwardObj::inferShape(const TensorVec &inputs) const {
    return {{inputs[0]->getDims()}};
}

std::string ActivationBackwardObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ActivationBackwardObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> ActivationBackwardObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

}; // namespace infini
