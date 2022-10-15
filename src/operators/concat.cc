#include "operators/concat.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int dim)
    : OperatorObj(OpType::Concat, inputs, {output}), dim(dim) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() > 1);
    Shape dims = inputs[0]->getDims();
    ShapeElem n = dims.at(dim);
    for (auto itr = inputs.begin() + 1; itr != inputs.end(); ++itr) {
        auto input = *itr;
        auto iDims = input->getDims();
        if (dims.size() != iDims.size())
            return {};
        int nDims = dims.size();
        for (auto i = 0; i < nDims; i++) {
            if (i == dim) {
                n += iDims.at(i);
                continue;
            }
            if (iDims.at(i) != dims.at(i))
                return {};
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
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> ConcatObj::getOpAttrVector() const {
    return {enum_to_underlying(type), dim};
}

} // namespace infini
