#include "operators/transpose.h"

namespace infini {
TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                           const Shape &perm)
    : OperatorObj(OpType::Transpose, {input}, {output}), perm(perm) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
TransposeObj::inferShape(const TensorVec &inputs) const {
    Shape dimsIn = inputs[0]->getDims();
    Shape dimsOut;
    std::unordered_set<size_t> dimSet;
    for (size_t i = 0; i < perm.size(); ++i) {
        if (size_t(perm[i]) >= dimsIn.size() ||
            dimSet.find(perm[i]) != dimSet.end()) {
            return {};
        }
        dimsOut.emplace_back(dimsIn[perm[i]]);
    }
    return {{dimsOut}};
}

std::string TransposeObj::toString() const {
    std::ostringstream os;
    os << "Transpose[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "perm=" << vecToString(perm) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> TransposeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), perm.begin(), perm.end());
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}
vector<int> TransposeObj::getOpAttrVector() const {
    vector<int> ret = perm;
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

} // namespace infini
