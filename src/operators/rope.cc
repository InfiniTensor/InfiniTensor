#include "operators/rope.h"

namespace infini {
RoPEObj::RoPEObj(GraphObj *graph, Tensor pos, Tensor input, Tensor output)
    : OperatorObj(OpType::RoPE, {pos, input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RoPEObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[1];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    return {{output_dim}};
}

std::string RoPEObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> RoPEObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> RoPEObj::getOpAttrVector() const { return {type.underlying()}; }

}; // namespace infini
