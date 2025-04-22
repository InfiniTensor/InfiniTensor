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
void ReduceBaseObj::initInfiniOp(const Runtime context) {
    auto input_dim = inputs[0]->getDims();
    auto output_dim = outputs[0]->getDims();

    // 转换为 infiniop 需要的 shape 格式
    auto input_shape = toInfiniopShape(input_dim);
    auto output_shape = toInfiniopShape(output_dim);
    auto opType=type;

    // 创建 tensor descriptor
    infiniopTensorDescriptor_t input_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &input_tensor, input_dim.size(), input_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));

    infiniopTensorDescriptor_t output_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &output_tensor, output_dim.size(), output_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    // 构造 axes 数组
    std::vector<int64_t> axes_vec(axes.begin(), axes.end());
    uint64_t n_axes = axes_vec.size();
    // int op_value;
    if(opType==OpType::ReduceMax){
        // op_value=0;
        CHECK_ERROR(infiniopCreateReduceMaxDescriptor(
            context->opHandle(),
            (infiniopReduceMaxDescriptor_t *)&opDesc,
            output_tensor,
            input_tensor,
            axes_vec.data(),
            n_axes,
            keepDims ? 1 : 0,
            0  // 可以根据实际需求改
        ));
    }else if(opType==OpType::ReduceMean){
        // op_value=1;
        CHECK_ERROR(infiniopCreateReduceMeanDescriptor(
            context->opHandle(),
            (infiniopReduceMeanDescriptor_t *)&opDesc,
            output_tensor,
            input_tensor,
            axes_vec.data(),
            n_axes,
            keepDims ? 1 : 0,
            0  // 可以根据实际需求改
        ));

    }else{
        // op_value=2;
        CHECK_ERROR(infiniopCreateReduceMinDescriptor(
            context->opHandle(),
            (infiniopReduceMinDescriptor_t *)&opDesc,
            output_tensor,
            input_tensor,
            axes_vec.data(),
            n_axes,
            keepDims ? 1 : 0,
            0  // 可以根据实际需求改
        ));

    }


    // // 创建 reduce descriptor
    // CHECK_ERROR(infiniopCreateReduceDescriptor(
    //     context->opHandle(),
    //     (infiniopReduceDescriptor_t *)&opDesc,
    //     output_tensor,
    //     input_tensor,
    //     axes_vec.data(),
    //     n_axes,
    //     keepDims ? 1 : 0,
    //     /*noop_with_empty_axes=*/0,  // 可以根据实际需求改
    //     static_cast<int>(opType)      // 假设 opType 映射为 reduce_type
    // ));

    // 销毁 tensor descriptors
    CHECK_ERROR(infiniopDestroyTensorDescriptor(input_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(output_tensor));
}


ReduceMeanObj::ReduceMeanObj(GraphObj *graph, Tensor input, Tensor output,
                             const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceMean, input, output, _axes, keepDims) {
}
ReduceMaxObj::ReduceMaxObj(GraphObj *graph, Tensor input, Tensor output,
    const optional<vector<int>> &_axes, bool keepDims)
: ReduceBaseObj(graph, OpType::ReduceMean, input, output, _axes, keepDims) {
}
ReduceMinObj::ReduceMinObj(GraphObj *graph, Tensor input, Tensor output,
    const optional<vector<int>> &_axes, bool keepDims)
: ReduceBaseObj(graph, OpType::ReduceMean, input, output, _axes, keepDims) {
}

ReduceSumObj::ReduceSumObj(GraphObj *graph, Tensor input, Tensor output,
                           const optional<vector<int>> &_axes, bool keepDims)
    : ReduceBaseObj(graph, OpType::ReduceSum, input, output, _axes, keepDims) {}
} // namespace infini
