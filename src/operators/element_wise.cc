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
    int max_len = std::max(A->getDims().size(), B->getDims().size());
    std::vector<int> A_(max_len, 1);
    std::vector<int> B_(max_len, 1);
    std::vector<int> res(max_len, 1);
    memcpy(A_.data() + max_len - A->getDims().size(), A->getDims().data(),
           A->getDims().size() * sizeof(int));
    memcpy(B_.data() + max_len - B->getDims().size(), B->getDims().data(),
           B->getDims().size() * sizeof(int));
    // std::copy(A->getDims().begin(), A->getDims().end(), A_.begin() + (max_len
    // - A->getDims().size())); std::copy(B->getDims().begin(),
    // B->getDims().end(), B_.begin() + (max_len - B->getDims().size()));
    // std::copy(A->getDims().rbegin(), A->getDims().rend(), A_.rbegin());
    // std::copy(B->getDims().rbegin(), B->getDims().rend(), B_.rbegin());

    for (int i = 0; i < max_len; ++i) {
        if (A_[i] == B_[i] || (A_[i] == 1 || B_[i] == 1)) {
            res[i] = std::max(A_[i], B_[i]);
        } else {
            return {};
        }
    }

    return {{res}};
}

std::string ElementWiseObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
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
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> ElementWiseObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

MSELossObj::MSELossObj(GraphObj *graph, Tensor input0, Tensor input1,
                       Reduction reduction, Tensor output)
    : OperatorObj(OpType::MSELoss, {input0, input1}, {output}),
      reductionMode(reduction) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> MSELossObj::inferShape(const TensorVec &inputs) const {
    const auto A = inputs[0], B = inputs[1];
    if (A->getDims().size() != B->getDims().size() ||
        A->getDims() != B->getDims())
        return {};

    if (reductionMode == None) {
        return {{A->getDims()}};
    } else {
        Shape temp = {1};
        return {{temp}};
    }
}

std::string MSELossObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
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
    ret.emplace(ret.begin(), enum_to_underlying(type));
    return ret;
}

vector<int> MSELossObj::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

}; // namespace infini
