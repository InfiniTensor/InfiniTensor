#include "operators/extend.h"

namespace infini {

ExtendObj::ExtendObj(GraphObj *graph, Tensor input, Tensor output, int dim,
                     int num)
    : OperatorObj(OpType::Extend, {input}, {output}), dim(dim), num(num) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ExtendObj::inferShape(const TensorVec &inputs) const {
    auto ret = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();
    IT_ASSERT((size_t)dim < rank);
    ret[dim] = ret[dim] * (num + 1);
    return {{ret}};
}
std::string ExtendObj::toString() const {
    std::ostringstream os;
    os << "Extend[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "num=" << num << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ExtendObj::getWorkloadVector() const {
    vector<int> ret = inputs[0]->getDims();
    ret.emplace_back(dim);
    ret.emplace_back(num);
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ExtendObj::getOpAttrVector() const {
    return {type.underlying(), dim, num};
}

} // namespace infini
