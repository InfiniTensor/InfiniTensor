#include "operators/element_wise.h"

namespace infini {
ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ElementWiseObj::inferShape(const TensorVec &inputs) const {
    // For now,we only process the same dims here, broardcast will be considered
    // in the opt layer.
    const auto A = inputs[0], B = inputs[1];
    int max_len = std::max(A->getRank(), B->getRank());
    std::vector<int> A_(max_len, 1);
    std::vector<int> B_(max_len, 1);
    std::vector<int> res(max_len, 1);
    memcpy(A_.data() + max_len - A->getRank(), A->getDims().data(),
           A->getRank() * sizeof(int));
    memcpy(B_.data() + max_len - B->getRank(), B->getDims().data(),
           B->getRank() * sizeof(int));
    // std::copy(A->getDims().begin(), A->getDims().end(), A_.begin() + (max_len
    // - A->getRank())); std::copy(B->getDims().begin(),
    // B->getDims().end(), B_.begin() + (max_len - B->getRank()));
    // std::copy(A->getDims().rbegin(), A->getDims().rend(), A_.rbegin());
    // std::copy(B->getDims().rbegin(), B->getDims().rend(), B_.rbegin());

    for (int i = 0; i < max_len; ++i) {
        IT_ASSERT(A_[i] == B_[i] || A_[i] == 1 || B_[i] == 1);
        res[i] = std::max(A_[i], B_[i]);
    }

    return {{res}};
}

std::string ElementWiseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> ElementWiseObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ElementWiseObj::getOpAttrVector() const {
    return {type.underlying()};
}

MSELossObj::MSELossObj(GraphObj *graph, Tensor input0, Tensor input1,
                       Reduction reduction, Tensor output)
    : OperatorObj(OpType::MSELoss, {input0, input1}, {output}),
      reductionMode(reduction) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> MSELossObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0], B = inputs[1];
    IT_ASSERT(A->getRank() == B->getRank());
    IT_ASSERT(A->getDims() == B->getDims());

    if (reductionMode == None) {
        return {{A->getDims()}};
    } else {
        Shape temp = {1};
        return {{temp}};
    }
}

std::string MSELossObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// use output dim or inputs dim?
vector<int> MSELossObj::getWorkloadVector() const {
    vector<int> ret = outputs[0]->getDims();
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> MSELossObj::getOpAttrVector() const { return {type.underlying()}; }

}; // namespace infini
