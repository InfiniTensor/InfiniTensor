#include "operators/global_pool.h"

namespace infini {

GlobalPoolObj::GlobalPoolObj(GraphObj *graph, OpType optype, Tensor input,
                             Tensor output)
    : OperatorObj(optype, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> GlobalPoolObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];

    auto ret = input->getDims();
    for (size_t i = 2; i < ret.size(); i++) {
        ret[i] = 1;
    }
    return {{ret}};
}

std::string GlobalPoolObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> GlobalPoolObj::getWorkloadVector() const {
    return {type.underlying()};
}

vector<int> GlobalPoolObj::getOpAttrVector() const {
    return {type.underlying()};
}

}; // namespace infini
