#include "operators/gemm.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {

GemmObj::GemmObj(GraphObj *graph, Tensor A, Tensor B, Tensor Y, Tensor C,
                 float alpha, float beta, bool transA, bool transB)
    : OperatorObj(OpType::Gemm, C ? TensorVec{A, B, C} : TensorVec{A, B}, {Y}),
      alpha(alpha), beta(beta), transA(transA), transB(transB) {
    IT_ASSERT(checkValid(graph));
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

void GemmObj::initInfiniOp(const Runtime context) {
    auto a_dim = inputs[0]->getDims();
    auto b_dim = inputs[1]->getDims();
    auto c_dim = inputs.size() == 3 ? inputs[2]->getDims() : Shape{};
    auto y_dim = outputs[0]->getDims();

    // convert dim data to infiniop format
    auto y_shape = toInfiniopShape(y_dim);
    auto a_shape = toInfiniopShape(a_dim);
    auto b_shape = toInfiniopShape(b_dim);
    auto c_shape = toInfiniopShape(c_dim);
    // create tensor descriptor
    infiniopTensorDescriptor_t y_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_tensor, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t a_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &a_tensor, a_dim.size(), a_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t b_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &b_tensor, b_dim.size(), b_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    infiniopTensorDescriptor_t c_tensor;
    if (inputs.size() == 3) {
        CHECK_ERROR(infiniopCreateTensorDescriptor(
            &c_tensor, c_dim.size(), c_shape.data(), nullptr,
            toInfiniopDataLayout(inputs[2]->getDType().getIndex())));
    }
    // create op descriptor
    CHECK_ERROR(infiniopCreateGEMMDescriptor(
        context->opHandle(), (infiniopGEMMDescriptor_t *)&opDesc, y_tensor,
        a_tensor, b_tensor, c_tensor, alpha, beta, transA, transB));

    // destroy tensor descriptor and op descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(a_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(b_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(c_tensor));
}

optional<vector<Shape>> GemmObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    int m = transA ? shapeA[1] : shapeA[0];
    int n = transB ? shapeB[0] : shapeB[1];
    Shape ret = {m, n};
    return {{ret}};
}

vector<int> GemmObj::getWorkloadVector() const {
    return {type.underlying(), transA, transB};
}

vector<int> GemmObj::getOpAttrVector() const {
    return {type.underlying(), transA, transB};
}

} // namespace infini
