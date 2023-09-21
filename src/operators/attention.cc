#include "operators/attention.h"
#include "utils/operator_utils.h"

namespace infini {

AttentionObj::AttentionObj(GraphObj *graph, Tensor inputQ, Tensor inputK,
                           Tensor inputV, Tensor output)
    : OperatorObj(OpType::Attention, TensorVec{inputQ, inputK, inputV},
                  {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
AttentionObj::inferShape(const TensorVec &inputs) const {
    auto shapeQ = inputs[0]->getDims();
    auto shapeK = inputs[1]->getDims();
    auto shapeV = inputs[2]->getDims();
    auto retQK = infer_broadcast(shapeQ, shapeK);
    auto ret = infer_broadcast(retQK, shapeV);
    return {{ret}};
}

std::string AttentionObj::toString() const {
    std::ostringstream os;
    os << "Attention[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[2]->getDims()) << ",";
    os << "inputQ=" << inputs[0]->getGuid() << ",";
    os << "inputK=" << inputs[1]->getGuid() << ",";
    os << "inputV=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> AttentionObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> AttentionObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
