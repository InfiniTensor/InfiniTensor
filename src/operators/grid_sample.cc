#include "operators/grid_sample.h"

namespace infini {

GridSampleObj::GridSampleObj(GraphObj *graph, Tensor input, Tensor grid,
                             Tensor output, bool nearest, bool reflection,
                             bool alignCorners)
    : OperatorObj(OpType::GridSample, {std::move(input), std::move(grid)},
                  {std::move(output)}),
      nearest(nearest), reflection(reflection), alignCorners(alignCorners) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> GridSampleObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0]->getDims();
    const auto &grid = inputs[1]->getDims();
    if (input.size() != 4 || grid.size() != 4 || grid[3] != 2 ||
        input[0] != grid[0] || !(inputs[0]->getDType() == inputs[1]->getDType()))
        return std::nullopt;
    return vector<Shape>{{grid[0], grid[1], grid[2], input[3]}};
}

std::string GridSampleObj::toString() const { return "GridSample(NHWC)"; }

vector<int> GridSampleObj::getOpAttrVector() const {
    return {type.underlying(), nearest, reflection, alignCorners};
}

vector<int> GridSampleObj::getWorkloadVector() const {
    auto workload = getOpAttrVector();
    for (const auto &input : inputs) {
        const auto dims = input->getDims();
        workload.insert(workload.end(), dims.begin(), dims.end());
    }
    return workload;
}

} // namespace infini
