#include "operators/logical.h"
#include "core/runtime.h"
#include "utils/operator_utils.h"

namespace infini {

BinaryLogicalObj::BinaryLogicalObj(OpType type, GraphObj *g, Tensor in0,
                                   Tensor in1, Tensor out)
    : OperatorObj(type, {std::move(in0), std::move(in1)}, {std::move(out)}) {
    IT_ASSERT(checkValid(g));
}

optional<vector<Shape>> BinaryLogicalObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    auto res = infer_broadcast(A->getDims(), B->getDims());
    return {{res}};
}

std::string BinaryLogicalObj::toString() const {
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
vector<int> BinaryLogicalObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> BinaryLogicalObj::getOpAttrVector() const {
    return {type.underlying()};
}

UnaryLogicalObj::UnaryLogicalObj(OpType type, GraphObj *g, Tensor in,
                                 Tensor out)
    : OperatorObj(type, {std::move(in)}, {std::move(out)}) {
    IT_ASSERT(checkValid(g));
}

optional<vector<Shape>> UnaryLogicalObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}

std::string UnaryLogicalObj::toString() const {
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
vector<int> UnaryLogicalObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> UnaryLogicalObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini