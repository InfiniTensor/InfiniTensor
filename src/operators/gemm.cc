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
