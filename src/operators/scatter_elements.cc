#include "operators/scatter_elements.h"

namespace infini {

ScatterElementsObj::ScatterElementsObj(GraphObj *graph, Tensor input,
                                       Tensor indices, Tensor updates,
                                       Tensor output, int axis, std::string reduction)
    : OperatorObj(OpType::ScatterElements,
                  {std::move(input), std::move(indices), std::move(updates)},
                  {std::move(output)}),
      axis(axis), reduction(std::move(reduction)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ScatterElementsObj::inferShape(const TensorVec &inputs) {
    const auto data = inputs[0]->getDims(), indices = inputs[1]->getDims();
    const auto rank = static_cast<int>(data.size());
    const auto dim = axis < 0 ? axis + rank : axis;
    const auto dtype = inputs[1]->getDType();
    if (dim < 0 || dim >= rank || indices.size() != data.size() ||
        indices != inputs[2]->getDims() ||
        !(dtype == DataType::Int32 || dtype == DataType::Int64) ||
        !(inputs[0]->getDType() == inputs[2]->getDType()))
        return std::nullopt;
    for (int i = 0; i < rank; ++i) {
        if (i != dim && indices[i] > data[i])
            return std::nullopt;
    }
    return vector<Shape>{data};
}

std::string ScatterElementsObj::toString() const { return "ScatterElements(" + reduction + ")"; }

vector<int> ScatterElementsObj::getOpAttrVector() const {
    const int mode = reduction == "none" ? 0 : reduction == "add" ? 1 :
                     reduction == "max" ? 2 : reduction == "min" ? 3 : -1;
    return {type.underlying(), axis, mode};
}

vector<int> ScatterElementsObj::getWorkloadVector() const {
    auto workload = getOpAttrVector();
    for (const auto &input : inputs) {
        const auto dims = input->getDims();
        workload.insert(workload.end(), dims.begin(), dims.end());
    }
    return workload;
}

} // namespace infini
