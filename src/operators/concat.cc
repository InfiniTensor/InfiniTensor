#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();
    if (inputs.size() == 2) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i]->size() == 0) {
                return {{inputs[1 - i]->getDims()}};
            }
        }
    }
    ShapeElem n = dims.at(dim);
    for (auto itr = inputs.begin() + 1; itr != inputs.end(); ++itr) {
        auto input = *itr;
        auto iDims = input->getDims();
        IT_ASSERT(rank == input->getRank());
        for (auto i = 0; i < (int)rank; i++) {
            if (i == dim) {
                n += iDims.at(i);
                continue;
            }
            IT_ASSERT(iDims.at(i) == dims.at(i));
        }
    }
    dims[dim] = n;
    return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ConcatObj::getWorkloadVector() const {
    vector<int> ret = getOutput()->getDims();
    ret.emplace(ret.begin(), (int)inputs.size());
    ret.emplace(ret.begin(), dim);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ConcatObj::getOpAttrVector() const {
    return {type.underlying(), dim};
}

} // namespace infini
