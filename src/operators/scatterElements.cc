#include "operators/scatterElements.h"
#include "utils/operator_utils.h"
namespace infini {

ScatterElementsObj::ScatterElementsObj(GraphObj *graph, Tensor data,
                                       Tensor indices, Tensor updates,
                                       Tensor output, int _axis)
    : OperatorObj(OpType::ScatterElements, TensorVec{data, indices, updates},
                  {output}) {
    int rank = inputs[0]->getRank();
    axis = get_real_axis(_axis, rank);
    IT_ASSERT(checkValid(graph));
}
optional<vector<Shape>>
ScatterElementsObj::inferShape(const TensorVec &inputs) {
    auto inputDims = inputs[0]->getDims();
    vector<Shape> ret;
    Shape outShape = inputDims;

    ret.push_back(outShape); // outshape = data.shape

    return {ret};
}
vector<DataType>
ScatterElementsObj::inferDataType(const TensorVec &inputs) const {
    return {inputs[0]->getDType()};
}
std::string ScatterElementsObj::toString() const {
    std::ostringstream os;
    os << "ScatterElements[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "data=" << inputs[0]->getGuid() << ",";
    os << "indices=" << inputs[1]->getGuid() << ",";
    os << "updates=" << inputs[2]->getGuid() << ",";
    os << "axis=" << axis << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}
vector<int> ScatterElementsObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());

    return ret;
}

vector<int> ScatterElementsObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

} // namespace infini
