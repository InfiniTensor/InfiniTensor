#include "operators/gemm.h"

namespace infini {

GemmObj::GemmObj(GraphObj *graph, Tensor A, Tensor B, Tensor Y, Tensor C,
                 float alpha, float beta, bool transA, bool transB)
    : OperatorObj(OpType::Gemm, TensorVec{A, B}, {Y}), alpha(alpha), beta(beta),
      transA(transA), transB(transB) {
    IT_ASSERT(checkValid(graph));
    if (C) {
        Y->copyin(C->getRawDataPtr<void *>(), C->getBytes());
    }
}

string GemmObj::toString() const {
    std::ostringstream os;
    os << "Gemm([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B")
       << "],A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
       << ",C="
       << (inputs.size() == 3 ? std::to_string(inputs[2]->getGuid()) : "null")
       << ",Y=" << outputs[0]->getGuid();
    return os.str();
}

optional<vector<Shape>> GemmObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    ShapeElem m = transA ? shapeA[1] : shapeA[0];
    ShapeElem n = transB ? shapeB[0] : shapeB[1];
    Shape ret = {m, n};
    return {{ret}};
}

vector<int> GemmObj::getWorkloadVector() const {
    return {type.underlying(), transA, transB};
}

vector<int> GemmObj::getOpAttrVector() const {
    return {type.underlying(), transA, transB};
}

void GemmObj::createOpDesc() {
    auto aShape = inputs[0]->getDims();
    auto bShape = inputs[1]->getDims();
    auto yShape = outputs[0]->getDims();
    auto aStride = inputs[0]->getStride();
    auto bStride = inputs[1]->getStride();
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
    // create gemm op descriptor
    CHECK_INFINI_ERROR(infiniopCreateGemmDescriptor(
        handle, (infiniopGemmDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
        bTensor));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(aTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
    CHECK_INFINI_ERROR(infiniopDestroyHandle(handle));
}

} // namespace infini