#include "operators/element_wise.h"
#include "utils/operator_utils.h"

namespace infini {
ElementWiseObj::ElementWiseObj(OpType type, GraphObj *graph, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type, {input0, input1}, {output}) {
    IT_ASSERT(checkValid(graph));
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
    vector<size_t> dims = outputs[0]->getDims();
    vector<int> ret(dims.begin(), dims.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> ElementWiseObj::getOpAttrVector() const {
    return {type.underlying()};
}

void ElementWiseObj::createOpDesc() {
    auto yShape = outputs[0]->getDims();
    auto aShape = yShape;
    auto bShape = yShape;
    auto aStride = inputs[0]->getStride();
    for (int i = yShape.size() - aStride.size(); i > 0; --i) {
        aStride.insert(aStride.begin(), 0);
    }
    auto bStride = inputs[1]->getStride();
    for (int i = yShape.size() - bStride.size(); i > 0; --i) {
        bStride.insert(bStride.begin(), 0);
    }
    auto yStride = outputs[0]->getStride();
    infiniopTensorDescriptor_t yTensor, aTensor, bTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape.size(), yShape.data(), yStride.data(),
        toInfiniDtype(outputs[0]->getDType())));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &aTensor, aShape.size(), aShape.data(), aStride.data(),
        toInfiniDtype(inputs[0]->getDType())));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &bTensor, bShape.size(), bShape.data(), bStride.data(),
        toInfiniDtype(inputs[1]->getDType())));
    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    if (type == OpType::Add) {
        CHECK_INFINI_ERROR(infiniopCreateAddDescriptor(
            handle, (infiniopAddDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
            bTensor));
    } else if (type == OpType::Mul) {
        CHECK_INFINI_ERROR(infiniopCreateMulDescriptor(
            handle, (infiniopMulDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
            bTensor));
    } else if (type == OpType::Sub) {
        CHECK_INFINI_ERROR(infiniopCreateSubDescriptor(
            handle, (infiniopSubDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
            bTensor));
    } else {
        IT_TODO_HALT_MSG("ElementWise operator not supported yet");
    }
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(aTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
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
    vector<size_t> dims = outputs[0]->getDims();
    vector<int> ret(dims.begin(), dims.end());
    ret.emplace(ret.begin(), type.underlying());
    return ret;
}

vector<int> MSELossObj::getOpAttrVector() const { return {type.underlying()}; }

}; // namespace infini
