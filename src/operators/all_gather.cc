#include "operators/all_gather.h"

namespace infini {
AllGatherObj::AllGatherObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::AllGather, {input}, {output}) {}

std::string AllGatherObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    return os.str();
}

vector<int> AllGatherObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = inputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> AllGatherObj::getOpAttrVector() const {
    return {type.underlying()};
}
} // namespace infini