#include "operators/logic.h"
#include "utils/operator_utils.h"

namespace infini {
LogicObj::LogicObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LogicObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    auto res = infer_broadcast(A->getDims(), B->getDims());
    return {{res}};
}

std::string LogicObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> LogicObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> LogicObj::getOpAttrVector() const {
    return {type.underlying()};
}
}; // namespace infini
