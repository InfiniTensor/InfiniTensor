#include "operators/expand.h"
#include "utils/operator_utils.h"

namespace infini {

ExpandObj::ExpandObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Expand, {input}, {output}), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

ExpandObj::ExpandObj(GraphObj *graph, Tensor input, Tensor shape, Tensor output)
    : OperatorObj(OpType::Expand, {input, shape}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ExpandObj::inferShape(const TensorVec &inputs) {
    if (inputs.size() == 2) {
        IT_ASSERT(inputs[1]->getShapeValue().has_value(),
                  "the target shape of this Expand is not known while shapes "
                  "are inferred, so it holds data rather than dimensions");
        dims = inputs[1]->getShapeValueAsShape();
    }
    auto shape_input = inputs[0]->getDims();
    Shape ret = infer_broadcast(shape_input, dims);
    return {{ret}};
}

vector<DimSource> ExpandObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output == 0);
    IT_ASSERT(dim < outputs[0]->getRank());
    const size_t outRank = outputs[0]->getRank();
    // Broadcasting lines the shapes up at the right, so one output dimension
    // sits at a different index in a shorter shape. `inferShape` leaves `dims`
    // holding the target of either form, so both are read the same way here.
    const size_t targetLead = outRank - dims.size();
    const size_t inputLead = outRank - inputs[0]->getRank();
    // A target read from an edge is a number only for the shape the graph
    // currently holds. Where that number may still change, it may turn out to
    // settle this dimension or to leave it to the input, and which of the two
    // it will be is not known yet -- so neither is claimed.
    if (inputs.size() == 2 && dim >= targetLead &&
        !inputs[1]->isShapeValueFixed(dim - targetLead)) {
        return OperatorObj::dimSources(output, dim);
    }
    // A target reaching this dimension and asking more than one of it settles
    // it: broadcasting only ever agrees with that number or stretches a one up
    // to it, so no input shape moves it.
    if (dim >= targetLead && dims[dim - targetLead] > 1) {
        return {};
    }
    // Left of the input's first dimension there is none to follow, and the
    // target asks only one there, so the answer is that one.
    if (dim < inputLead) {
        return {};
    }
    return {DimSource{0, dim - inputLead}};
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
