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
    // get dim data
    auto input_dim = inputs[0]->getDims();
    auto output_dim = outputs[0]->getDims();

    // convert dim data to infiniop format
    auto input_shape = toInfiniopShape(input_dim);
    auto output_shape = toInfiniopShape(output_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t input_desc, output_desc;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &input_desc, input_dim.size(), input_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &output_desc, output_dim.size(), output_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));
    
    // create op descriptor
    vector<int64_t> axes_vec;
    for (auto it : axes) {
        axes_vec.emplace_back(it);
    }

    if (type == OpType::ReduceMax) {
        CHECK_ERROR(infiniopCreateReduceMaxDescriptor(
            context->opHandle(), (infiniopReduceMaxDescriptor_t *)&opDesc,
            output_desc, input_desc, axes_vec.data(), axes_vec.size(),
            keepDims, 0));
    } else if (type == OpType::ReduceMean) {
        CHECK_ERROR(infiniopCreateReduceMeanDescriptor(
            context->opHandle(), (infiniopReduceMeanDescriptor_t *)&opDesc,
            output_desc, input_desc, axes_vec.data(), axes_vec.size(),
            keepDims, 0));
    } else if (type == OpType::ReduceMin) {
        CHECK_ERROR(infiniopCreateReduceMinDescriptor(
            context->opHandle(), (infiniopReduceMinDescriptor_t *)&opDesc,
            output_desc, input_desc, axes_vec.data(), axes_vec.size(),
            keepDims, 0));
    }

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(input_desc));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(output_desc));
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

ReduceMaxObj::ReduceMaxObj(GraphObj *graph, Tensor input, Tensor output,
    const optional<vector<int>> &_axes, bool keepDims)
: ReduceBaseObj(graph, OpType::ReduceMax, input, output, _axes, keepDims) {
}

ReduceMeanObj::ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                             const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMean, input, output, _axes, keepDims) {
}

ReduceMinObj::ReduceMinObj(GraphObj *graph, Tensor input, Tensor output,
    const optional<vector<int>> &_axes, bool keepDims)
: ReduceBaseObj(graph, OpType::ReduceMin, input, output, _axes, keepDims) {
}

ReduceSumObj::ReduceSumObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceSum, input, output, _axes, keepDims) {}
} // namespace infini
