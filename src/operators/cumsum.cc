#include "operators/unary.h"

namespace infini {

CumsumObj::CumsumObj(GraphObj *graph, Tensor input, Tensor output, int axis,
              bool exclusive, bool reverse)
    : OperatorObj(OpType::CumSum, {input}, {output}),
        axisValue(axis), exclusiveValue(exclusive), reverseValue(reverse) {
        IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> CumsumObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    return {{A->getDims()}};
}

std::string CumsumObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> CumsumObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> CumsumObj::getOpAttrVector() const { return {type.underlying()}; }

};