#include "operators/sendrecv.h"

namespace infini {
SendRecvObj::SendRecvObj(GraphObj *graph, Tensor input, Tensor output,
                         int source, int destination, Shape dims)
    : OperatorObj(OpType::SendRecv, {input}, {output}), source(source),
      destination(destination), dims(std::move(dims)) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SendRecvObj::inferShape(const TensorVec &inputs) {

    size_t size = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        size *= dims.at(i);
    }
    IT_ASSERT(size == inputs[0]->size());

    return {{dims}};
}

vector<DataType> SendRecvObj::inferDataType(const TensorVec &inputs) const {
    return {{inputs[0]->getDType()}};
}

std::string SendRecvObj::toString() const {
    std::ostringstream os;
    os << "SendRecv"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "dims=" << vecToString(dims) << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SendRecvObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.insert(ret.end(), dims.begin(), dims.end());
    ret.emplace(ret.begin(), type.underlying());

    ret.emplace_back(source);
    ret.emplace_back(destination);

    return ret;
}

vector<int> SendRecvObj::getOpAttrVector() const {
    vector<int> ret = dims;
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);
    return ret;
}
} // namespace infini
