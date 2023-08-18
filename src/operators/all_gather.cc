#include "operators/all_gather.h"

namespace infini {
AllGatherObj::AllGatherObj(GraphObj *graph, Tensor input,
                           std::optional<TensorVec> outputs, int world_size)
    : OperatorObj(
          OpType::AllGather, {input},
          ((!outputs) ? TensorVec(world_size, nullptr) : std::move(*outputs))),
      world_size(world_size) {
    // if (!outputs) {
    //     TensorVec tmp(world_size, nullptr);
    //     this->outputs = tmp;
    // }
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
AllGatherObj::inferShape(const TensorVec &inputs) const {
    Shape input_shape = inputs[0]->getDims();
    vector<Shape> output_shapes(getWorldSize(), input_shape);
    return output_shapes;
}

vector<DataType> AllGatherObj::inferDataType(const TensorVec &inputs) const {
    return vector<DataType>(world_size, inputs[0]->getDType());
}

std::string AllGatherObj::toString() const {
    std::ostringstream os;
    os << "AllGather"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=";
    for (auto i = 0; i < world_size; i++)
        os << outputs[i]->getGuid() << ",";
    os << ")";
    return os.str();
}

vector<int> AllGatherObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = inputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    ret.emplace_back(world_size);
    return ret;
}

vector<int> AllGatherObj::getOpAttrVector() const {
    return {type.underlying(), world_size};
}
} // namespace infini