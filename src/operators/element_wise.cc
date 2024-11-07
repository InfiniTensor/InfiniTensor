#include "operators/element_wise.h"
#include "utils/operator_utils.h"

namespace infini {
ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
}

void ElementWiseObj::initInfiniOp(const Runtime context) {
    auto a_dim = inputs[0]->getDims();
    auto b_dim = inputs[1]->getDims();
    auto c_dim = outputs[0]->getDims();

    if (type == OpType::Add) {
        auto a_shape = toInfiniopShape(a_dim);
        auto b_shape = toInfiniopShape(b_dim);
        auto c_shape = toInfiniopShape(c_dim);
        // create tensor descriptor
        infiniopTensorDescriptor_t a_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &a_tensor, a_dim.size(), a_shape.data(), nullptr,
            toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
        infiniopTensorDescriptor_t b_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &b_tensor, b_dim.size(), b_shape.data(), nullptr,
            toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
        infiniopTensorDescriptor_t c_tensor;
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &c_tensor, c_dim.size(), c_shape.data(), nullptr,
            toInfiniopDataLayout(outputs[0]->getDType().getIndex())));
        // create op descriptor
        CHECK_ERROR(infiniopCreateAddDescriptor(
            context->opHandle(), (infiniopAddDescriptor_t *)&opDesc, c_tensor,
            a_tensor, b_tensor));

        // destroy tensor descriptor and op descriptor
        CHECK_ERROR(infiniopDestroyTensorDescriptor(a_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(b_tensor));
        CHECK_ERROR(infiniopDestroyTensorDescriptor(c_tensor));
    } else {
        opDesc = nullptr;
    }
}

optional<vector<Shape>> ElementWiseObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0], B = inputs[1];
    auto res = infer_broadcast(A->getDims(), B->getDims());
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

optional<vector<Shape>> MSELossObj::inferShape(const TensorVec &inputs) {
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
