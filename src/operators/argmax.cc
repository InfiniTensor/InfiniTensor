#include "operators/argmax.h"
#include "core/graph.h"
#include "cuda/cuda_runtime.h"

namespace infini {

ArgMaxObj::ArgMaxObj(GraphObj *graph, Tensor input, Tensor output, int axis,
                     bool keepDims, bool selectLastIndex)
    : OperatorObj(OpType::ArgMax, {input}, {output}),
      axis(axis), keepDims(keepDims), selectLastIndex(selectLastIndex) {
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

std::string ArgMaxObj::toString() const {
    return "ArgMax(axis=" + std::to_string(axis) + 
           ", keepDims=" + std::to_string(keepDims) + 
           ", selectLastIndex=" + std::to_string(selectLastIndex) + ")";
}

vector<int> ArgMaxObj::getWorkloadVector() const {
    // 参考 AttentionKVCacheObj 的实现方式
    vector<int> ret = getOutputs()[0]->getDims();
    ret.emplace(ret.begin(), (int)selectLastIndex);
    ret.emplace(ret.begin(), (int)keepDims);
    ret.emplace(ret.begin(), axis);
    return ret;
}

vector<int> ArgMaxObj::getOpAttrVector() const {
    // 返回算子的关键属性，用于唯一标识算子配置
    return {type.underlying()};
}

} // namespace infini