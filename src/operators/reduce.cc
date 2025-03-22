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

void ReduceBaseObj::initInfiniOp(const Runtime context) {
    auto data_dim = inputs[0]->getDims();
    auto reduced_dim = outputs[0]->getDims();

    auto x_shape = toInfiniopShape(data_dim);
    auto y_shape = toInfiniopShape(reduced_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t data_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &data_tensor, data_dim.size(), x_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t reduced_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &reduced_tensor, reduced_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    // get const int64_t * of axes
    std::vector<int64_t> axes_;
    axes_.reserve(axes.size());
    for (int axis : axes) {
        axes_.push_back(static_cast<int64_t>(axis));
    }
    const int64_t *axesPtr = axes_.data();

    // create op descriptor
    if (type == OpType::ReduceMin) {
        CHECK_ERROR(infiniopCreateReduceMinDescriptor(
            context->opHandle(), (infiniopReduceMinDescriptor_t *)&opDesc,
            reduced_tensor, data_tensor,
            (int64_t *)axesPtr, axes.size(),
            keepDims, noop_with_empty_axes));
    } else if (type == OpType::ReduceMax) {
        CHECK_ERROR(infiniopCreateReduceMaxDescriptor(
            context->opHandle(), (infiniopReduceMaxDescriptor_t *)&opDesc,
            reduced_tensor, data_tensor,
            (int64_t *)axesPtr, axes.size(),
            keepDims, noop_with_empty_axes));
    } else if (type == OpType::ReduceMean) {
        CHECK_ERROR(infiniopCreateReduceMeanDescriptor(
            context->opHandle(), (infiniopReduceMeanDescriptor_t *)&opDesc,
            reduced_tensor, data_tensor,
            (int64_t *)axesPtr, axes.size(),
            keepDims, noop_with_empty_axes));
    } else {
        IT_TODO_HALT();
    }

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(reduced_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(data_tensor));
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
        if (ret.empty())
            return {{{1}}};
        else
            return {{ret}};
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

ReduceMinObj::ReduceMinObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMin, input, output, _axes, keepDims) {
}

ReduceMaxObj::ReduceMaxObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMax, input, output, _axes, keepDims) {
}

ReduceMeanObj::ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                             const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMean, input, output, _axes, keepDims) {
}

ReduceSumObj::ReduceSumObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceSum, input, output, _axes, keepDims) {}
} // namespace infini
