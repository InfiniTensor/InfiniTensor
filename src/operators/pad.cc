#include "operators/pad.h"

namespace infini {
PadObj::PadObj(GraphObj *graph, Tensor input, Tensor output,
               const vector<int> &_pads,
               const optional<const vector<int>> &axis)
    : OperatorObj(OpType::Pad, {input}, {output}) {
    if (axis == std::nullopt)
        pads = _pads;
    else {
        int nAxis = (*axis).size();
        IT_ASSERT((int)_pads.size() == nAxis * 2);
        int nDims = input->getDims().size();
        vector<int> tmp(nDims * 2, 0);

        for (int i = 0; i < nAxis; ++i) {
            tmp[(*axis)[i]] = _pads[i];
            tmp[(*axis)[i] + nDims] = _pads[i + nAxis];
        }
        pads = tmp;
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PadObj::inferShape(const TensorVec &inputs) const {
    auto dims = inputs[0]->getDims();
    int nDims = dims.size();
    if (nDims * 2 != (int)pads.size())
        return {};
    for (int i = 0; i < nDims; ++i) {
        if (pads[i] < 0 || pads[i + nDims] < 0)
            return {};
        dims[i] += pads[i] + pads[i + nDims];
    }

    return {{dims}};
}
std::string PadObj::toString() const {
    std::ostringstream os;
    os << "Pad"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "pads=" << vecToString(pads) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PadObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), pads.begin(), pads.end());
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> PadObj::getOpAttrVector() const {
    vector<int> ret = pads;
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

} // namespace infini
