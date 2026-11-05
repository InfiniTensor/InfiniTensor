#include "operators/scatter_nd.h"

namespace infini {

ScatterNdObj::ScatterNdObj(GraphObj *graph, Tensor input, Tensor indices,
                             Tensor updates, Tensor output, bool add)
    : OperatorObj(OpType::ScatterND,
                  {std::move(input), std::move(indices), std::move(updates)},
                  {std::move(output)}),
      add(add) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ScatterNdObj::inferShape(const TensorVec &inputs) {
    const auto &data = inputs[0]->getDims();
    const auto &indices = inputs[1]->getDims();
    const auto &updates = inputs[2]->getDims();
    const auto indexType = inputs[1]->getDType();
    if (indices.empty() || indices.back() <= 0 ||
        static_cast<size_t>(indices.back()) > data.size() ||
        !(indexType == DataType::Int64 || indexType == DataType::Int32) ||
        !(inputs[0]->getDType() == inputs[2]->getDType()))
        return std::nullopt;
    Shape expected(indices.begin(), indices.end() - 1);
    expected.insert(expected.end(), data.begin() + indices.back(), data.end());
    if (expected != updates)
        return std::nullopt;
    return vector<Shape>{data};
}

std::string ScatterNdObj::toString() const { return add ? "ScatterND(add)" : "ScatterND"; }

vector<int> ScatterNdObj::getOpAttrVector() const {
    return {type.underlying(), add};
}

vector<int> ScatterNdObj::getWorkloadVector() const {
    auto workload = getOpAttrVector();
    for (const auto &input : inputs) {
        const auto dims = input->getDims();
        workload.insert(workload.end(), dims.begin(), dims.end());
    }
    return workload;
}

} // namespace infini
