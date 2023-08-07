#include "operators/softmax.h"

namespace infini {

SoftmaxObj::SoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int _axis)
    : OperatorObj(OpType::Softmax, {input}, {output}) {
    if (_axis >= 0 && (size_t)_axis < input->getRank())
        axis = _axis;
    else if (_axis <= -1 && (size_t)_axis >= -input->getRank())
        axis = _axis + input->getRank();
    else
        IT_ASSERT(0);
    IT_ASSERT(checkValid(graph));
}

std::string SoftmaxObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "axis=" << axis << ")";
    return os.str();
}

vector<int> SoftmaxObj::getWorkloadVector() const {
    vector<int> ret{type.underlying(), axis};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> SoftmaxObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}
} // namespace infini
