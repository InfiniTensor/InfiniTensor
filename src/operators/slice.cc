#include "operators/slice.h"

namespace infini {
SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor output,
                   const vector<int> &starts, const vector<int> &ends,
                   const optional<vector<int>> &axis,
                   const optional<vector<int>> &steps)
    : OperatorObj(OpType::Slice, {input}, {output}) {
    if (steps != std::nullopt)
        IT_TODO_HALT();
    IT_ASSERT(starts.size() == ends.size());

    if (axis == std::nullopt) {
        this->starts = starts;
        this->ends = ends;
    } else {
        int nAxis = (*axis).size();
        IT_ASSERT((int)starts.size() == nAxis);

        int nDims = input->getDims().size();
        vector<int> tmpS(nDims, 0), tmpE;
        for (int i = 0; i < nDims; ++i) {
            tmpE.emplace_back(input->getDims()[i] - 1);
        }

        for (int i = 0; i < nAxis; ++i) {
            if ((*axis)[i] < 0)
                IT_TODO_HALT();
            tmpS[(*axis)[i]] = starts[i];
            tmpE[(*axis)[i]] = ends[i];
        }
        this->starts = tmpS;
        this->ends = tmpE;
    }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SliceObj::inferShape(const TensorVec &inputs) const {
    auto dims = inputs[0]->getDims();
    int nDims = dims.size();
    if (nDims != (int)starts.size())
        return {};
    for (int i = 0; i < nDims; ++i) {
        if (starts[i] < 0 || ends[i] >= dims[i] || starts[i] > ends[i])
            return {};
        dims[i] = ends[i] - starts[i] + 1;
    }

    return {{dims}};
}

std::string SliceObj::toString() const {
    std::ostringstream os;
    os << "Slice"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "starts=" << vecToString(starts) << ",";
    os << "ends=" << vecToString(ends) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SliceObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), starts.begin(), starts.end());
    ret.insert(ret.end(), ends.begin(), ends.end());
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> SliceObj::getOpAttrVector() const {
    vector<int> ret = starts;
    ret.insert(ret.end(), ends.begin(), ends.end());
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

} // namespace infini
