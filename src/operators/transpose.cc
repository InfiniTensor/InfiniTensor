#include "operators/transpose.h"

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           vector<int> permute)
    : OperatorObj(OpType::Transpose, {input}, {output}) {
    if (permute.size() != 4) {
        IT_TODO_HALT();
    }
    transposePermute[0] = permute[0];
    transposePermute[1] = permute[1];
    transposePermute[2] = permute[2];
    transposePermute[3] = permute[3];
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
TransposeObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    auto input = A->getDims();
    auto output = input;

    for (int i = 0; i < 4; ++i) {
        output[i] = input[transposePermute[i]];
    }
    return {{output}};
}

std::string TransposeObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> TransposeObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> TransposeObj::getOpAttrVector() const {
    return {type.underlying()};
}

}; // namespace infini
