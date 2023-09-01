#include "operators/transpose.h"

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           vector<int> permute)
    : OperatorObj(OpType::Transpose, {input}, {output}) {
    auto rank = input->getRank();
    if (permute.empty()) {
        for (size_t i = 0; i < rank; ++i) {
            transposePermute[i] = i;
        }
    } else {
        IT_ASSERT(rank == permute.size());
        transposePermute = std::move(permute);
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    int rank = A->getRank();

    for (auto index : transposePermute) {
        IT_ASSERT(index < rank);
    }
    for (int i = 0; i < rank; ++i) {
        output_dim[i] = input_dim[transposePermute[i]];
    }
    return {{output_dim}};
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
