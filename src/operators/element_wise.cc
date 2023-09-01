#include "operators/element_wise.h"
#include "utils/operator_utils.h"

namespace infini {
ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ElementWiseObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    auto res = infer_broadcast(A->getDims(), B->getDims());
    return {{res}};
}

std::string ElementWiseObj::toString() const {
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
vector<int> ElementWiseObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ElementWiseObj::getOpAttrVector() const {
    return {type.underlying()};
}

MSELossObj::MSELossObj(GraphObj *graph, Tensor input0, Tensor input1,
                       Reduction reduction, Tensor output)
    : OperatorObj(OpType::MSELoss, {input0, input1}, {output}),
      reductionMode(reduction) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> MSELossObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    IT_ASSERT(A->getRank() == B->getRank());
    IT_ASSERT(A->getDims() == B->getDims());

    if (reductionMode == None) {
        return {{A->getDims()}};
    } else {
        Shape temp = {1};
        return {{temp}};
    }
}

std::string MSELossObj::toString() const {
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
vector<int> MSELossObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> MSELossObj::getOpAttrVector() const { return {type.underlying()}; }

}; // namespace infini
