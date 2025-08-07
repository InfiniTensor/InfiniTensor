#include "operators/send.h"

namespace infini {
SendObj::SendObj(GraphObj *graph, Tensor input, int source, int destination,
                 [[maybe_unused]] Tensor output)
    : OperatorObj(OpType::Send, TensorVec{input},
                  TensorVec{output ? output : nullptr}),
      source(source), destination(destination) {

    IT_ASSERT(checkValid(graph));
}
optional<vector<Shape>> SendObj::inferShape(const TensorVec &inputs) {
    return {{inputs[0]->getDims()}};
}
vector<DataType> SendObj::inferDataType(const TensorVec &inputs) const {
    return {{inputs[0]->getDType()}};
}

std::string SendObj::toString() const {
    std::ostringstream os;
    os << "Send"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> SendObj::getWorkloadVector() const {
    vector<size_t> tmp = inputs[0]->getDims();
    vector<int> ret(tmp.begin(), tmp.end());
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);

    return ret;
}

vector<int> SendObj::getOpAttrVector() const {
    vector<size_t> tmp = inputs[0]->getDims();
    vector<int> ret(tmp.begin(), tmp.end());
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);
    return ret;
}
} // namespace infini
