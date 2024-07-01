#include "operators/softmax.h"
#include "utils/operator_utils.h"

namespace infini {

SoftmaxObj::SoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int _axis)
    : OperatorObj(OpType::Softmax, {input}, {output}) {
    int rank = input->getRank();
    axis = get_real_axis(_axis, rank);
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
BangSoftmaxObj::BangSoftmaxObj(GraphObj *graph, Tensor input, Tensor output,
                               int _axis)
    : OperatorObj(OpType::BangSoftmax, {input}, {output}) {
    int rank = input->getRank();
    axis = get_real_axis(_axis, rank);
    IT_ASSERT(checkValid(graph));
}

std::string BangSoftmaxObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "axis=" << axis << ")";
    return os.str();
}

vector<int> BangSoftmaxObj::getWorkloadVector() const {
    vector<int> ret{type.underlying(), axis};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> BangSoftmaxObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

} // namespace infini
