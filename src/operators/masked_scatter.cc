#include "operators/masked_scatter.h"

namespace infini {

MaskedScatterObj::MaskedScatterObj(GraphObj *graph, Tensor input, Tensor mask,
                                   Tensor source, Tensor output)
    : OperatorObj(OpType::MaskedScatter,
                  {std::move(input), std::move(mask), std::move(source)},
                  {std::move(output)}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> MaskedScatterObj::inferShape(const TensorVec &inputs) {
    const auto data = inputs[0]->getDims();
    const auto mask = inputs[1]->getDims();
    if (mask.size() > data.size() || !(inputs[1]->getDType() == DataType::Bool) ||
        !(inputs[0]->getDType() == inputs[2]->getDType()))
        return std::nullopt;
    const auto offset = data.size() - mask.size();
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] != 1 && mask[i] != data[offset + i])
            return std::nullopt;
    }
    return vector<Shape>{data};
}

std::string MaskedScatterObj::toString() const { return "MaskedScatter"; }

vector<int> MaskedScatterObj::getOpAttrVector() const {
    return {type.underlying()};
}

vector<int> MaskedScatterObj::getWorkloadVector() const {
    auto workload = getOpAttrVector();
    for (const auto &input : inputs) {
        const auto dims = input->getDims();
        workload.insert(workload.end(), dims.begin(), dims.end());
    }
    return workload;
}

} // namespace infini
