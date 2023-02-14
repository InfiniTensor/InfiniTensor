#include "operators/slice.h"

namespace infini {
SliceObj::SliceObj(GraphObj *graph, Tensor input, Tensor output,
                   const vector<int> &starts, const vector<int> &ends,
                   const optional<vector<int>> &axes,
                   const optional<vector<int>> &steps)
    : OperatorObj(OpType::Slice, {input}, {output}) {
    if (steps)
        IT_TODO_HALT();
    IT_ASSERT(starts.size() == ends.size());

    if (!axes) {
        this->starts = starts;
        this->ends = ends;
    } else {
        auto nAxis = (*axes).size();
        IT_ASSERT(starts.size() == nAxis);

        auto dims = input->getDims();
        this->starts = vector<int>(dims.size(), 0);
        this->ends.resize(dims.size());
        std::transform(dims.begin(), dims.end(), this->ends.begin(),
                       [](auto x) { return x - 1; });

        for (size_t j = 0; j < nAxis; ++j) {
            auto i = (*axes)[j];
            if (i < 0)
                IT_TODO_HALT();
            this->starts[i] = starts[j];
            this->ends[i] = ends[j];
        }
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
