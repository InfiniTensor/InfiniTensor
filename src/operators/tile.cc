#include "operators/tile.h"

#include <limits>

namespace infini {

TileObj::TileObj(GraphObj *graph, Tensor input, Tensor output, Shape repeats)
    : OperatorObj(OpType::Tile, {std::move(input)}, {std::move(output)}),
      repeats(std::move(repeats)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> TileObj::inferShape(const TensorVec &inputs) {
    auto shape = inputs[0]->getDims();
    if (shape.size() != repeats.size())
        return std::nullopt;
    for (size_t i = 0; i < shape.size(); ++i) {
        const auto size = int64_t{shape[i]} * repeats[i];
        if (repeats[i] < 0 || size < 0 || size > std::numeric_limits<int>::max())
            return std::nullopt;
        shape[i] = static_cast<int>(size);
    }
    return vector<Shape>{shape};
}

std::string TileObj::toString() const { return "Tile"; }

vector<int> TileObj::getOpAttrVector() const {
    auto attrs = repeats;
    attrs.insert(attrs.begin(), type.underlying());
    return attrs;
}

vector<int> TileObj::getWorkloadVector() const {
    auto workload = getOpAttrVector();
    const auto shape = inputs[0]->getDims();
    workload.insert(workload.end(), shape.begin(), shape.end());
    return workload;
}

} // namespace infini
