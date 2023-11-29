#include "operators/expand.h"
#include "utils/operator_utils.h"

namespace infini {

ExpandObj::ExpandObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Expand, {input}, {output}), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ExpandObj::inferShape(const TensorVec &inputs) {
    auto shape_input = inputs[0]->getDims();
    Shape ret = infer_broadcast(shape_input, dims);
    return {{ret}};
}

std::string ExpandObj::toString() const {
    std::ostringstream os;
    os << "Expand[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "dims=" << vecToString(dims) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ExpandObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), dims.begin(), dims.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ExpandObj::getOpAttrVector() const {
    vector<int> ret = dims;
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

} // namespace infini
