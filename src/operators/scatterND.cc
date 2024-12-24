#include "operators/scatterND.h"
#include "utils/operator_utils.h"
namespace infini {

ScatterNDObj::ScatterNDObj(GraphObj *graph, Tensor data, Tensor indices,
                           Tensor updates, Tensor output)
    : OperatorObj(OpType::ScatterND, TensorVec{data, indices, updates},
                  {output}) {
    IT_ASSERT(checkValid(graph));
}
optional<vector<Shape>> ScatterNDObj::inferShape(const TensorVec &inputs) {
    auto inputDims = inputs[0]->getDims();
    vector<Shape> ret;
    Shape outShape = inputDims;

    ret.push_back(outShape); // outshape = data.shape

    return {ret};
}
vector<DataType> ScatterNDObj::inferDataType(const TensorVec &inputs) const {
    return {inputs[0]->getDType()};
}
std::string ScatterNDObj::toString() const {
    std::ostringstream os;
    os << "ScatterND[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "data=" << inputs[0]->getGuid() << ",";
    os << "indices=" << inputs[1]->getGuid() << ",";
    os << "updates=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}
vector<int> ScatterNDObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());

    return ret;
}

vector<int> ScatterNDObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini

