#include "operators/matmul.h"

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
    auto rank = std::max(rankA, rankB);
    if (rankA < rank) {
        for (int i = 0; i < rank - rankA; ++i) {
            shape_a.insert(shape_a.begin(), 1);
        }
    }
    if (rankB < rank) {
        for (int i = 0; i < rank - rankB; ++i) {
            shape_b.insert(shape_b.begin(), 1);
        }
    }
    b = 1;
    for (int i = 0; i < rank - 2; ++i) {
        b *= std::max(shape_a[i], shape_b[i]);
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
    auto rank = std::max(rankA, rankB);
    if (rankA < rank) {
        for (int i = 0; i < rank - rankA; ++i) {
            shapeA.insert(shapeA.begin(), 1);
        }
    }
    if (rankB < rank) {
        for (int i = 0; i < rank - rankB; ++i) {
            shapeB.insert(shapeB.begin(), 1);
        }
    }
    Shape ret;
    for (int i = 0; i < rank - 2; ++i) {
        IT_ASSERT(shapeA[i] == shapeB[i] || shapeA[i] == 1 || shapeB[i] == 1);
        auto shape_ele = std::max(shapeA[i], shapeB[i]);
        ret.emplace_back(shape_ele);
    }
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
