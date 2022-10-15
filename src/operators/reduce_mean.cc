#include "operators/reduce_mean.h"

namespace infini {
ReduceMeanObj::ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                             const optional<const vector<int>> &_axis,
                             bool keepDims)
    : OperatorObj(OpType::ReduceMean, {input}, {output}), keepDims(keepDims) {

    if (_axis != std::nullopt) {
        IT_ASSERT((*_axis).size() <= input->getDims().size());
        for (size_t j = 0; j < (*_axis).size(); ++j) {
            int idx = (*_axis)[j];
            if (idx < 0)
                IT_TODO_HALT();
            IT_ASSERT((size_t)idx < input->getDims().size());
            axis.emplace(idx);
        }
    } else
        for (size_t i = 0; i < input->getDims().size(); ++i)
            axis.emplace(i);
    IT_ASSERT(checkValid(graph));
}

bool ReduceMeanObj::isReduced(int idx) const {
    return axis.find(idx) != axis.end();
}

optional<vector<Shape>>
ReduceMeanObj::inferShape(const TensorVec &inputs) const {
    auto dims = inputs[0]->getDims();

    if (keepDims) {
        Shape ret = dims;
        for (auto it : axis)
            ret[it] = 1;
        return {{ret}};
    } else {
        Shape ret;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (!isReduced(i))
                ret.emplace_back(dims[i]);
        }
        if (ret.size() == (size_t)0)
            ret.emplace_back(1);
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
    for (auto d : axis) {
        axisstr.append(std::to_string(d));
        axisstr.append(",");
    }
    if (!axis.empty())
        axisstr.pop_back();
    axisstr.append("]");
    os << "axis=" << axisstr << ",";
    os << "keepDims=" << keepDims << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReduceMeanObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    ret.emplace_back((int)keepDims);
    ret.insert(ret.end(), axis.begin(), axis.end());
    return ret;
}

vector<int> ReduceMeanObj::getOpAttrVector() const {
    vector<int> ret = {enum_to_underlying(type), (int)keepDims};
    ret.insert(ret.end(), axis.begin(), axis.end());
    return ret;
}
} // namespace infini
