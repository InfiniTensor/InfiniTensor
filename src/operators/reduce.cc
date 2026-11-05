#include "operators/reduce.h"
#include "utils/operator_utils.h"

namespace infini {
ReduceBaseObj::ReduceBaseObj(GraphObj *graph, OpType opType, Tensor input,
                             Tensor output, const optional<vector<int>> &_axes,
                             bool keepDims)
    : OperatorObj(opType, {input}, {output}), keepDims(keepDims) {
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

bool ReduceBaseObj::isReduced(int idx) const {
    return axes.find(idx) != axes.end();
}

optional<vector<Shape>> ReduceBaseObj::inferShape(const TensorVec &inputs) {
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
        return vector<Shape>{ret};
    }
}

std::string ReduceBaseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
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

vector<int> ReduceBaseObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back((int)keepDims);
    ret.insert(ret.end(), axes.begin(), axes.end());
    return ret;
}

vector<int> ReduceBaseObj::getOpAttrVector() const {
    vector<int> ret = {type.underlying(), (int)keepDims};
    ret.insert(ret.end(), axes.begin(), axes.end());
    return ret;
}

ReduceMeanObj::ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                             const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMean, input, output, _axes, keepDims) {
}

ReduceSumObj::ReduceSumObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceSum, input, output, _axes, keepDims) {}

ReduceMaxObj::ReduceMaxObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMax, input, output, axes, keepDims) {}

ReduceL2Obj::ReduceL2Obj(GraphObj *graph, Tensor input, Tensor output,
                        const optional<vector<int>> &axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceL2, input, output, axes, keepDims) {}

ArgMaxObj::ArgMaxObj(GraphObj *graph, Tensor input, Tensor output, int axis,
                     bool keepDims)
    : OperatorObj(OpType::ArgMax, {input}, {output}),
      axis(get_real_axis(axis, input->getRank())), keepDims(keepDims) {
    IT_ASSERT(!output || output->getDType() == DataType::Int64);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ArgMaxObj::inferShape(const TensorVec &inputs) {
    auto shape = inputs[0]->getDims();
    IT_ASSERT(axis >= 0 && static_cast<size_t>(axis) < shape.size());
    IT_ASSERT(shape[axis] > 0, "ArgMax reduction axis must not be empty");
    if (keepDims)
        shape[axis] = 1;
    else
        shape.erase(shape.begin() + axis);
    return vector<Shape>{shape};
}

vector<DataType> ArgMaxObj::inferDataType(const TensorVec &) const {
    return {DataType::Int64};
}

std::string ArgMaxObj::toString() const {
    return "ArgMax[" + std::to_string(getGuid()) + "](axis=" +
           std::to_string(axis) + ",keepDims=" + std::to_string(keepDims) + ")";
}

vector<int> ArgMaxObj::getOpAttrVector() const {
    return {type.underlying(), axis, static_cast<int>(keepDims)};
}

vector<int> ArgMaxObj::getWorkloadVector() const {
    auto workload = getOpAttrVector();
    const auto dims = inputs[0]->getDims();
    workload.insert(workload.end(), dims.begin(), dims.end());
    return workload;
}
} // namespace infini
