#include "operators/roi_align.h"

#include <cstring>

namespace infini {

RoiAlignObj::RoiAlignObj(GraphObj *graph, Tensor data, Tensor boxes, Tensor output,
                         int height, int width, int samplingRatio, float spatialScale,
                         bool aligned)
    : OperatorObj(OpType::RoiAlign, {std::move(data), std::move(boxes)},
                  {std::move(output)}),
      height(height), width(width), samplingRatio(samplingRatio),
      spatialScale(spatialScale), aligned(aligned) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RoiAlignObj::inferShape(const TensorVec &inputs) {
    const auto &data = inputs[0]->getDims();
    const auto &boxes = inputs[1]->getDims();
    if (data.size() != 4 || boxes.size() != 2 || boxes[1] != 5 ||
        height <= 0 || width <= 0 || samplingRatio < 0 ||
        !(inputs[0]->getDType() == inputs[1]->getDType()))
        return std::nullopt;
    return vector<Shape>{{boxes[0], height, width, data[3]}};
}

std::string RoiAlignObj::toString() const { return "RoiAlign(NHWC)"; }

vector<int> RoiAlignObj::getOpAttrVector() const {
    static_assert(sizeof(int) == sizeof(float));
    int scaleBits = 0;
    std::memcpy(&scaleBits, &spatialScale, sizeof(scaleBits));
    return {type.underlying(), height, width, samplingRatio, scaleBits, aligned};
}

vector<int> RoiAlignObj::getWorkloadVector() const {
    auto workload = getOpAttrVector();
    for (const auto &input : inputs) {
        const auto dims = input->getDims();
        workload.insert(workload.end(), dims.begin(), dims.end());
    }
    return workload;
}

} // namespace infini
