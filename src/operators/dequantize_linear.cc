#include "operators/dequantize_linear.h"
#include "utils/operator_utils.h"

namespace infini {

DequantizeLinearObj::DequantizeLinearObj(GraphObj *graph, Tensor inputX,
                                         Tensor inputScale, Tensor output,
                                         [[maybe_unused]] Tensor inputZeroPoint,
                                         int axis)
    : OperatorObj(OpType::DequantizeLinear,
                  inputZeroPoint ? TensorVec{inputX, inputScale, inputZeroPoint}
                                 : TensorVec{inputX, inputScale},
                  {output}),
      axis(axis) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
DequantizeLinearObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}}; // x.shape = output.shape = inputs[0].shape
}
vector<DataType>
DequantizeLinearObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 2 || inputs.size() == 3);

    return {
        inputs[1]->getDType()}; // scale.dtype = output.dtype = inputs[1].dtype
}

std::string DequantizeLinearObj::toString() const {
    std::ostringstream os;
    os << "DequantizeLinear[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "inputX=" << inputs[0]->getGuid() << ",";
    os << "inputScale=" << inputs[1]->getGuid() << ",";
    // os << "inputZeroPoint=" << inputs[2]->getGuid() << ",";
    os << "axis=" << axis << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> DequantizeLinearObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> DequantizeLinearObj::getOpAttrVector() const {
    return {type.underlying(), axis};
}

} // namespace infini
