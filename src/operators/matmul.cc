#include "operators/matmul.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                     bool transB, [[maybe_unused]] Tensor bias, ActType act,
                     std::string computeType)
    : OperatorObj(OpType::MatMul,
                  bias ? TensorVec{A, B, bias} : TensorVec{A, B}, {C}),
      transA(transA), transB(transB), act(act), b(1), computeType(computeType) {
    IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
    std::ostringstream os;
    os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B")
       << ",act=" << enum_to_underlying(act) << "],A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ",bmnk=[" << b << "," << m << "," << n << "," << k << "])"
       << ",computeType=" << computeType;
    return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    int rankA = A->getRank(); // Rank is the Shape of TensorDims
    int rankB = B->getRank();
    Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
    Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));
    Shape ret = infer_broadcast(shapeA1, shapeB1);
    if (ret.empty()) {
        b = 1;
    } else {
        b = std::accumulate(ret.begin(), ret.end(), 1, std::multiplies<int>());
    }
    auto kA = *(transA ? shapeA.rbegin() + 1 : shapeA.rbegin());
    auto kB = *(transB ? shapeB.rbegin() : shapeB.rbegin() + 1);
    IT_ASSERT(kA == kB);
    m = *(transA ? shapeA.rbegin() : shapeA.rbegin() + 1);
    n = *(transB ? shapeB.rbegin() + 1 : shapeB.rbegin());
    k = kA;
    ret.emplace_back(m);
    ret.emplace_back(n);
    return {{ret}};
}

vector<DimSource> MatmulObj::dimSources(size_t output, size_t dim) const {
    IT_ASSERT(output == 0);
    const auto rank = outputs[0]->getRank();
    IT_ASSERT(dim < rank);
    // The result is [..., M, N]. K constrains validity, but does not size an
    // output axis; neither does a bias broadcast onto that result.
    if (dim == rank - 2) {
        const auto rankA = inputs[0]->getRank();
        return {{0, rankA - (transA ? 1 : 2)}};
    }
    if (dim == rank - 1) {
        const auto rankB = inputs[1]->getRank();
        return {{1, rankB - (transB ? 2 : 1)}};
    }
    // Batch axes align from the right, just as in inferShape. A dynamic axis
    // currently equal to one must remain a source: it can grow on the next
    // request and become the side that determines the broadcast extent.
    vector<DimSource> sources;
    for (size_t i = 0; i < 2; ++i) {
        const auto missing = rank - inputs[i]->getRank();
        if (dim >= missing) {
            sources.push_back({i, dim - missing});
        }
    }
    return sources;
}

vector<int> MatmulObj::getWorkloadVector() const {
    return {type.underlying(),      b, m, n, k, transA, transB,
            enum_to_underlying(act)};
}

vector<int> MatmulObj::getOpAttrVector() const {
    return {type.underlying(), transA, transB, enum_to_underlying(act)};
}

} // namespace infini
