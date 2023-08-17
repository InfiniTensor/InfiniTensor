#include "operators/reduce_mean.h"
#include "utils/operator_utils.h"

namespace infini {
ReduceMeanObj::ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                             const optional<vector<int>> &_axes, bool keepDims)
    : OperatorObj(OpType::ReduceMean, {input}, {output}), keepDims(keepDims) {
    const auto size = input->getRank();
    if (_axes) {
        for (auto idx : *_axes) {
            idx = get_real_axis(idx, size);
            axes.emplace(idx);
        }
    } else
        for (size_t i = 0; i < size; ++i)
            axes.emplace(i);
    IT_ASSERT(checkValid(graph));
}

bool ReduceMeanObj::isReduced(int idx) const {
    return axes.find(idx) != axes.end();
}

optional<vector<Shape>>
ReduceMeanObj::inferShape(const TensorVec &inputs) const {
    auto dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    if (keepDims) {
        Shape ret = dims;
        for (auto it : axes)
            ret[it] = 1;
        return {{ret}};
    } else {
        Shape ret;
        for (size_t i = 0; i < rank; ++i) {
            if (!isReduced(i))
                ret.emplace_back(dims[i]);
        }
        if (ret.empty())
            return {{{1}}};
        else
            return {{ret}};
    }
}

std::string ReduceMeanObj::toString() const {
    std::ostringstream os;
    os << "ReduceMean"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";

    std::string axisstr;
    axisstr.append("[");
    for (auto d : axes) {
        axisstr.append(std::to_string(d));
        axisstr.append(",");
    }
    if (!axes.empty())
        axisstr.pop_back();
    axisstr.append("]");
    os << "axes=" << axisstr << ",";
    os << "keepDims=" << keepDims << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReduceMeanObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back((int)keepDims);
    ret.insert(ret.end(), axes.begin(), axes.end());
    return ret;
}

vector<int> ReduceMeanObj::getOpAttrVector() const {
    vector<int> ret = {type.underlying(), (int)keepDims};
    ret.insert(ret.end(), axes.begin(), axes.end());
    return ret;
}
} // namespace infini
