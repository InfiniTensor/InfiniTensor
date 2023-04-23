#include "operators/reshape.h"

namespace infini {
ReshapeObj::ReshapeObj(GraphObj *graph, Tensor input, Tensor output, Shape dims)
    : OperatorObj(OpType::Reshape, {input}, {output}),
      dims(dims.size() == 0 ? output->getDims() : dims) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ReshapeObj::inferShape(const TensorVec &inputs) const {
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); ++i)
        size *= dims.at(i);
    if (size != inputs[0]->size())
        return {};

    return {{dims}};
}

std::string ReshapeObj::toString() const {
    std::ostringstream os;
    os << "Reshape[" << getGuid() << "]";
    os << "(input dim=";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "output dims=" << vecToString(dims) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ReshapeObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), dims.begin(), dims.end());
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}
vector<int> ReshapeObj::getOpAttrVector() const {
    vector<int> ret = dims;
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

FlattenObj::FlattenObj(GraphObj *graph, Tensor input, Tensor output, int _axis)
    : OperatorObj(OpType::Flatten, {input}, {output}) {
    if (_axis >= 0 && (size_t)_axis < input->getDims().size())
        axis = _axis;
    else if (_axis <= -1 && (size_t)_axis >= -input->getDims().size())
        axis = _axis + input->getDims().size();
    else
        IT_ASSERT(0);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> FlattenObj::inferShape(const TensorVec &inputs) const {
    int sizeB = 1, sizeE = 1;
    auto dims = getInputs(0)->getDims();
    int ndim = dims.size();
    for (int i = 0; i < ndim; ++i)
        ((i < axis) ? sizeB : sizeE) *= dims.at(i);

    return {{{sizeB, sizeE}}};
}

std::string FlattenObj::toString() const {
    std::ostringstream os;
    os << "Flatten[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "axis=" << axis << ")";
    return os.str();
}

vector<int> FlattenObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), axis);
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> FlattenObj::getOpAttrVector() const {
    return {enum_to_underlying(type), axis};
}

IdentityObj::IdentityObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::Identity, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> IdentityObj::inferShape(const TensorVec &inputs) const {
    return {{getInputs(0)->getDims()}};
}

std::string IdentityObj::toString() const {
    std::ostringstream os;
    os << "Identity[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> IdentityObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}
vector<int> IdentityObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}
} // namespace infini
