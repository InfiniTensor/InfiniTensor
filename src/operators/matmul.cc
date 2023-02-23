#include "operators/matmul.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB, [[maybe_unused]] Tensor bias, ActType act)
    : OperatorObj(OpType::Matmul, {A, B}, {C}), transA(transA), transB(transB),
      act(act), b(1) {
    auto shape_a = A->getDims();
    auto shape_b = B->getDims();
    IT_ASSERT(shape_a.size() == shape_b.size());
    switch (shape_a.size()) {
    case 0:
    case 1:
        IT_ASSERT(false);
    case 2:
        break;
    default:
        for (size_t i = 0; i < shape_a.size() - 2; ++i) {
            IT_ASSERT(shape_a[i] == shape_b[i]);
            b *= shape_a[i];
        }
        break;
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
    auto shape_a = inputs[0]->getDims();
    auto it = shape_a.rbegin();
    *it++ = n;
    *it++ = m;
    return {{std::move(shape_a)}};
}

vector<int> MatmulObj::getWorkloadVector() const {
    return {enum_to_underlying(type), b, m, n, k, transA, transB,
            enum_to_underlying(act)};
}

vector<int> MatmulObj::getOpAttrVector() const {
    return {enum_to_underlying(type), transA, transB, enum_to_underlying(act)};
}

} // namespace infini
