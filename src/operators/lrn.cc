#include "operators/lrn.h"
#include "utils/operator_utils.h"

namespace infini {

LRNObj::LRNObj(GraphObj *graph, Tensor input, Tensor output, float alpha,
               float beta, float bias, int size)
    : OperatorObj(OpType::LRN, TensorVec{input}, {output}), alpha_value(alpha),
      beta_value(beta), bias_value(bias), size_value(size) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> LRNObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string LRNObj::toString() const {
    std::ostringstream os;
    os << "LRN[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> LRNObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> LRNObj::getOpAttrVector() const { return {type.underlying()}; }

} // namespace infini
