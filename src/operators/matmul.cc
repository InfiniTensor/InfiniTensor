#include "operators/matmul.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB, [[maybe_unused]] Tensor bias, ActType act)
    : OperatorObj(OpType::Matmul,
                  bias ? TensorVec{A, B, bias} : TensorVec{A, B}, {C}),
      transA(transA), transB(transB), act(act), b(1) {
    auto shape_a = A->getDims();
    auto shape_b = B->getDims();
    int dimA = shape_a.size(), dimB = shape_b.size();
    IT_ASSERT(dimA >= 2 && dimB >= 2);

    b = 1;
    if (dimA <= 3 && dimB <= 3) {
        int b1 = dimA == 2 ? 1 : A->getDims()[0];
        int b2 = dimB == 2 ? 1 : B->getDims()[0];

        b = std::max(b1, b2);
    } else {
        IT_ASSERT_TODO(dimA == dimB);
        for (size_t i = 0; i < shape_a.size() - 2; ++i) {
            IT_ASSERT_TODO(shape_a[i] == shape_b[i]);
            b *= shape_a[i];
        }
    }
    m = *(transA ? shape_a.rbegin() : shape_a.rbegin() + 1);
    n = *(transB ? shape_b.rbegin() + 1 : shape_b.rbegin());
    k = *(transA ? shape_a.rbegin() + 1 : shape_a.rbegin());
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
    int dimA = A->getDims().size(), dimB = B->getDims().size();

    if (dimA > 3 || dimB > 3) {
        // no broadcast
        auto shape_a = inputs[0]->getDims();
        auto it = shape_a.rbegin();
        *it++ = n;
        *it++ = m;
        return {{std::move(shape_a)}};
    }

    int b1 = dimA == 2 ? 1 : A->getDims()[0];
    int b2 = dimB == 2 ? 1 : B->getDims()[0];

    int b = std::max(b1, b2);
    int m = transA ? A->getDims()[dimA - 1] : A->getDims()[dimA - 2];
    int n = transB ? B->getDims()[dimB - 2] : B->getDims()[dimB - 1];
    int kA = transA ? A->getDims()[dimA - 2] : A->getDims()[dimA - 1];
    int kB = transB ? B->getDims()[dimB - 1] : B->getDims()[dimB - 2];

    if ((dimA != 2 && dimA != 3) || (dimB != 2 && dimB != 3)) {
        printf("Bad input dim: dimA = %d, dimB = %d\n", dimA, dimB);
        return {};
    }
    if (b1 != 1 && b2 != 1 && b1 != b2) {
        printf("Bad batch size b1 = %d, b2 = %d\n", b1, b2);
        return {};
    }
    if (kA != kB) {
        printf("Bad K: kA = %d, kB = %d\n", kA, kB);
        return {};
    }
    if (dimA == 2 && dimB == 2) {
        return {{{m, n}}};
    } else {
        return {{{b, m, n}}};
    }
}

vector<int> MatmulObj::getWorkloadVector() const {
    return {enum_to_underlying(type), b, m, n, k, transA, transB,
            enum_to_underlying(act)};
}

vector<int> MatmulObj::getOpAttrVector() const {
    return {enum_to_underlying(type), transA, transB, enum_to_underlying(act)};
}

} // namespace infini
