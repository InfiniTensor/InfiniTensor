#include "operators/recv.h"

namespace infini {
RecvObj::RecvObj(GraphObj *graph, Tensor output, int source, int destination,
                 Shape dims, int outputType, [[maybe_unused]] Tensor input)
    : OperatorObj(OpType::Recv, TensorVec{input ? input : nullptr},
                  TensorVec{output}),
      source(source), destination(destination), dims(std::move(dims)),
      outputType(outputType) {

    IT_ASSERT(checkValid(graph));
}
optional<vector<Shape>> RecvObj::inferShape(const TensorVec &inputs) {
    return {{dims}};
}
vector<DataType> RecvObj::inferDataType(const TensorVec &inputs) const {
    return {{DataType(outputType)}};
}

std::string RecvObj::toString() const {
    std::ostringstream os;
    os << "Recv"
       << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(dims) << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "dims=" << vecToString(dims) << ")";
    return os.str();
}

vector<int> RecvObj::getWorkloadVector() const {
    vector<int> ret = dims;
    ret.insert(ret.end(), dims.begin(), dims.end());
    ret.emplace(ret.begin(), type.underlying());

    ret.emplace_back(source);
    ret.emplace_back(destination);

    return ret;
}

vector<int> RecvObj::getOpAttrVector() const {
    vector<int> ret = dims;
    ret.emplace(ret.begin(), type.underlying());
    ret.emplace_back(source);
    ret.emplace_back(destination);
    return ret;
}
} // namespace infini
