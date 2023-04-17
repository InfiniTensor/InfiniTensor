#include "operators/unsqueeze.h"

namespace infini {
UnsqueezeObj::UnsqueezeObj(GraphObj *graph, Tensor in,
                           const std::vector<int> &index, Tensor out)
    : OperatorObj(OpType::Unsqueeze, {in}, {out}) {
    IT_ASSERT(parseAxis(index, axis));
    IT_ASSERT(checkValid(graph));
}

bool UnsqueezeObj::parseAxis(const std::vector<int> &index,
                             std::set<int> &axis) const {
    bool ret = true;
    int nDim = inputs[0]->getDims().size() + index.size();
    for (size_t i = 0; i < index.size(); ++i) {
        int data = index[i];
        if (data < 0)
            data += nDim;
        if (data >= nDim) {
            ret = false;
            break;
        }
        if (axis.find(data) != axis.end()) {
            ret = false;
            break;
        }
        axis.insert(data);
    }
    return ret;
}

optional<vector<Shape>>
UnsqueezeObj::inferShape(const TensorVec &inputs) const {
    Shape dims = inputs[0]->getDims();
    for (int i : axis) {
        auto it = dims.begin();
        dims.insert(std::next(it, i), 1);
    }
    return {{dims}};
}

std::string UnsqueezeObj::toString() const {
    std::ostringstream os;
    os << "Unsqueeze[" << getGuid() << "]";
    os << "(";
    os << "inputs=";
    for (auto i = 0; i < numInputs(); i++)
        os << inputs[i]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << ")";

    return os.str();
}

vector<int> UnsqueezeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    for (auto i : axis)
        ret.emplace_back(i);
    return ret;
}
vector<int> UnsqueezeObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}
} // namespace infini
