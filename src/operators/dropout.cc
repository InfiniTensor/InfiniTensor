#include "operators/dropout.h"

namespace infini {

DropoutObj::DropoutObj(GraphObj *graph, Tensor data, Tensor output, Tensor mask,
                       float ratio, bool training_mode)
    : OperatorObj(OpType::Dropout, {data}, {output, mask}), ratio(ratio) {
    IT_ASSERT(0 <= ratio && ratio < 1);
    IT_ASSERT(!training_mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> DropoutObj::inferShape(const TensorVec &inputs) {
    auto shape = inputs[0]->getDims();
    return {{shape, shape}};
}
std::string DropoutObj::toString() const {
    std::ostringstream os;
    os << "Dropout[" << getGuid() << "](" << vecToString(inputs[0]->getDims())
       << ", "
       << "ratio=" << ratio << ", "
       << "training_mode=false, "
       << "input=" << inputs[0]->getGuid() << ", "
       << "outputs=" << outputs[0]->getGuid() << ", " << outputs[1]->getGuid()
       << ")";
    return os.str();
}

vector<int> DropoutObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace_back(static_cast<int>(ratio));
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> DropoutObj::getOpAttrVector() const {
    return {type.underlying(), static_cast<int>(ratio), false};
}

} // namespace infini
