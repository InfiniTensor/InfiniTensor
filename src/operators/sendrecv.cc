#include "operators/sendrecv.h"

namespace infini {
SendRecvObj::SendRecvObj(GraphObj *graph, Tensor input, Tensor output,
                         int source, int destination)
    : OperatorObj(OpType::SendRecv, {input}, {output}), source(source),
      destination(destination) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> SendRecvObj::inferShape(const TensorVec &inputs) const {

    return {{inputs[0]->getDims()}};
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
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SendRecvObj::getWorkloadVector() const {
    vector<int> ret{type.underlying()};
    const Shape shape = inputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    ret.emplace_back(source);
    ret.emplace_back(destination);

    return ret;
}

vector<int> SendRecvObj::getOpAttrVector() const {
    return {type.underlying(), source, destination};
}
} // namespace infini
