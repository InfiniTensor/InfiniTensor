#include "operators/matmul.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB, [[maybe_unused]] Tensor bias, ActType act)
    : OperatorObj(OpType::MatMul,
                  bias ? TensorVec{A, B, bias} : TensorVec{A, B}, {C}),
      transA(transA), transB(transB), act(act), b(1) {
    auto shape_a = A->getDims();
    auto shape_b = B->getDims();
    int rankA = A->getRank();
    int rankB = B->getRank();
    IT_ASSERT(rankA >= 2 && rankB >= 2);
    Shape shape_a1(shape_a.begin(), shape_a.begin() + (rankA - 2));
    Shape shape_b1(shape_b.begin(), shape_b.begin() + (rankB - 2));
    auto ret = infer_broadcast(shape_a1, shape_b1);
    if (ret.empty()) {
        b = 1;
    } else {
        b = std::accumulate(ret.begin(), ret.end(), 1, std::multiplies<int>());
    }
    auto kA = *(transA ? shape_a.rbegin() + 1 : shape_a.rbegin());
    auto kB = *(transB ? shape_b.rbegin() : shape_b.rbegin() + 1);
    IT_ASSERT(kA == kB);
    m = *(transA ? shape_a.rbegin() : shape_a.rbegin() + 1);
    n = *(transB ? shape_b.rbegin() + 1 : shape_b.rbegin());
    k = kA;
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B")
       << ",act=" << enum_to_underlying(act) << "],A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ",bmnk=[" << b << "," << m << "," << n << "," << k << "])";
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) const {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    int rankA = A->getRank();
    int rankB = B->getRank();
    Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
    Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));
    Shape ret = infer_broadcast(shapeA1, shapeB1);
    ret.emplace_back(m);
    ret.emplace_back(n);
    return {{ret}};
}

vector<int> MatmulObj::getWorkloadVector() const {
    return {type.underlying(),      b, m, n, k, transA, transB,
            enum_to_underlying(act)};
}

vector<int> MatmulObj::getOpAttrVector() const {
    return {type.underlying(), transA, transB, enum_to_underlying(act)};
}

} // namespace infini
