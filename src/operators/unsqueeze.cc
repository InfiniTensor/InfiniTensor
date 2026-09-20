#include "operators/unsqueeze.h"
#include "utils/operator_utils.h"

namespace infini {
UnsqueezeObj::UnsqueezeObj(GraphObj *graph, Tensor input, Tensor output,
                           Shape axes)
    : OperatorObj(OpType::Unsqueeze, {input}, {output}), axes(std::move(axes)) {
    IT_ASSERT(checkValid(graph));
}

UnsqueezeObj::UnsqueezeObj(GraphObj *graph, Tensor input, Tensor axesTensor,
                           Tensor output)
    : OperatorObj(OpType::Unsqueeze, {input, axesTensor}, {output}),
      dynamicAxes(true) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> UnsqueezeObj::inferShape(const TensorVec &inputs) {
    Shape inputDim = inputs[0]->getDims();
    Shape currentAxes = axes;
    if (dynamicAxes && inputs[1]->hasData()) {
        currentAxes.clear();
        if (inputs[1]->getDType() == DataType::Int64)
            for (auto x : inputs[1]->copyout<int64_t>())
                currentAxes.push_back(static_cast<int>(x));
        else {
            IT_ASSERT(inputs[1]->getDType() == DataType::Int32);
            for (auto x : inputs[1]->copyout<int32_t>())
                currentAxes.push_back(x);
        }
    }
    auto rank = inputs[0]->getRank() + currentAxes.size();
    Shape outputShape(rank, -1);
    for (size_t i = 0; i < currentAxes.size(); ++i) {
        currentAxes[i] = get_real_axis(currentAxes[i], rank);
        IT_ASSERT(outputShape[currentAxes[i]] == -1, "Axes have duplicate");
        outputShape[currentAxes[i]] = 1;
    }
    auto it = inputDim.begin();
    for (size_t i = 0; i < outputShape.size(); ++i) {
        if (outputShape[i] == -1) {
            outputShape[i] = *it++;
        }
    }
    return {{outputShape}};
}

std::string UnsqueezeObj::toString() const {
    std::ostringstream os;
    os << "Unsqueeze[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "axes=" << vecToString(axes) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> UnsqueezeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), axes.begin(), axes.end());
    if (dynamicAxes)
        ret.emplace_back(inputs[1]->getDims().size());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}
vector<int> UnsqueezeObj::getOpAttrVector() const {
    vector<int> ret = axes;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

} // namespace infini
