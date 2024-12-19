#include "operators/topk.h"
#include "utils/operator_utils.h"
namespace infini {

TopKObj::TopKObj(GraphObj *graph, Tensor input,
                 std::optional<TensorVec> outputs, Shape K, int axis,
                 int Largest, int sorted)
    : OperatorObj(OpType::TopK, {input},
                  ((!outputs) ? TensorVec(2, nullptr) : std::move(*outputs))),
      K(std::move(K)), axis(axis), Largest(Largest), sorted(sorted) {
    IT_ASSERT(checkValid(graph));
}
optional<vector<Shape>> TopKObj::inferShape(const TensorVec &inputs) {
    // k's size must be 1
    IT_ASSERT(K.size() == 1);
    auto inputDims = inputs[0]->getDims();
    vector<Shape> ret;
    Shape outShape = inputDims;

    for (int i = 0; i < 2; i++) {
        outShape[axis] = K[0];
        ret.push_back(outShape);
    }

    return {ret};
}
vector<DataType> TopKObj::inferDataType(const TensorVec &inputs) const {
    return {inputs[0]->getDType(), DataType(7)}; // 7表示int64
}
std::string TopKObj::toString() const {
    std::ostringstream os;
    os << "TopK[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "K=" << K[0] << ",";
    os << "axis=" << axis << ",";
    os << "Largest=" << Largest << ",";
    os << "sorted=" << sorted << ",";
    os << "values=" << outputs[0]->getGuid() << ",";
    os << "Indices=" << outputs[1]->getGuid() << ")";
    return os.str();
}
vector<int> TopKObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(axis);
    ret.emplace_back(Largest);
    ret.emplace_back(sorted);

    return ret;
}

vector<int> TopKObj::getOpAttrVector() const {
    return {type.underlying(), axis, Largest, sorted};
}

} // namespace infini
