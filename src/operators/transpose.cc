#include "operators/transpose.h"

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           vector<int> permute)
    : OperatorObj(OpType::Transpose, {input}, {output}) {
    transposePermute = permute;
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
TransposeObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0];
    auto input = A->getDims();
    auto output = input;

    auto nDims = input.size();
    for (size_t i = 0; i < nDims; ++i) {
        output[i] = input[transposePermute[i]];
    }
    return {{output}};
}

std::string TransposeObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "perm=" << vecToString(transposePermute) << ")";
    return os.str();
}

vector<int> TransposeObj::getWorkloadVector() const {
    vector<int> ret{enum_to_underlying(type)};
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> TransposeObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

}; // namespace infini
