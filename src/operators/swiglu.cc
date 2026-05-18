#include "operators/swiglu.h"

namespace infini {
SwiGLUObj::SwiGLUObj(GraphObj *graph, Tensor input, Tensor gate, Tensor output)
    : OperatorObj(OpType::SwiGLU, {input, gate}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SwiGLUObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string SwiGLUObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "gate=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SwiGLUObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> SwiGLUObj::getOpAttrVector() const { return {type.underlying()}; }

} // namespace infini
