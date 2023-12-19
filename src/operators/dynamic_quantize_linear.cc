#include "operators/dynamic_quantize_linear.h"
#include "utils/operator_utils.h"

namespace infini {
DynamicQuantizeLinearObj::DynamicQuantizeLinearObj(
    GraphObj *graph, Tensor input, std::optional<TensorVec> outputs)
    : OperatorObj(OpType::DynamicQuantizeLinear, TensorVec{input},
                  ((!outputs) ? TensorVec(3, nullptr) : std::move(*outputs))) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
DynamicQuantizeLinearObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims(), {}, {}}};
}

vector<DataType>
DynamicQuantizeLinearObj::inferDataType(const TensorVec &inputs) const {
    IT_ASSERT(inputs.size() == 1);
    return {DataType(2), DataType(1), DataType(2)};
}

std::string DynamicQuantizeLinearObj::toString() const {
    std::ostringstream os;
    os << "DynamicQuantizeLinear[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=";
    for (auto output : outputs)
        os << output->getGuid() << ",";
    os << ")";
    return os.str();
}

vector<int> DynamicQuantizeLinearObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> DynamicQuantizeLinearObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
