#include "operators/argmax.h"

namespace infini {

ArgMaxObj::ArgMaxObj(GraphObj *graph, Tensor input, Tensor output, int axis,
                     int keepDims, int selectLastIndex)
    : OperatorObj(OpType::ArgMax, {input}, {output}), axis(axis),
      keepDims(keepDims), selectLastIndex(selectLastIndex) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ArgMaxObj::inferShape(const TensorVec &inputs) {
    auto input = inputs[0];
    auto inputShape = input->getDims();
    int rank = static_cast<int>(inputShape.size());

    // 处理负轴
    int realAxis = axis < 0 ? axis + rank : axis;
    IT_ASSERT(realAxis >= 0 && realAxis < rank);

    vector<Shape> outputShapes;
    if (keepDims) {
        Shape outputShape = inputShape;
        outputShape[realAxis] = 1;
        outputShapes.push_back(outputShape);
    } else {
        Shape outputShape;
        for (int i = 0; i < rank; i++) {
            if (i != realAxis) {
                outputShape.push_back(inputShape[i]);
            }
        }
        outputShapes.push_back(outputShape);
    }

    return outputShapes;
}

vector<DataType> ArgMaxObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 1);

    return {DataType::Int64};
}

std::string ArgMaxObj::toString() const {
    return "ArgMax(axis=" + std::to_string(axis) +
           ", keepDims=" + std::to_string(keepDims) +
           ", selectLastIndex=" + std::to_string(selectLastIndex) + ")";
}

vector<int> ArgMaxObj::getWorkloadVector() const {
    vector<int> ret = getOutputs()[0]->getDims();
    ret.emplace(ret.begin(), (int)selectLastIndex);
    ret.emplace(ret.begin(), (int)keepDims);
    ret.emplace(ret.begin(), axis);
    return ret;
}

vector<int> ArgMaxObj::getOpAttrVector() const { return {type.underlying()}; }

} // namespace infini