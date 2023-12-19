#include "operators/matmul_integer.h"
#include "utils/operator_utils.h"
#include <numeric>

namespace infini {

MatmulIntegerObj::MatmulIntegerObj(GraphObj *graph, Tensor A, Tensor B,
                                   Tensor C,
                                   [[maybe_unused]] Tensor a_zero_point,
                                   [[maybe_unused]] Tensor b_zero_point)
    : OperatorObj(OpType::MatMulInteger,
                  a_zero_point ? (b_zero_point ? TensorVec{A, B, a_zero_point,
                                                           b_zero_point}
                                               : TensorVec{A, B, a_zero_point})
                               : TensorVec{A, B},
                  {C}),
      b(1) {
    IT_ASSERT(checkValid(graph));
}

string MatmulIntegerObj::toString() const {
    std::ostringstream os;
    os << "MatmulInteger(A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ",bmnk=[" << b << "," << m << "," << n << "," << k << "])";
    return os.str();
}

optional<vector<Shape>> MatmulIntegerObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getDims();
    auto shapeB = B->getDims();
    int rankA = A->getRank();
    int rankB = B->getRank();
    Shape shapeA1(shapeA.begin(), shapeA.begin() + (rankA - 2));
    Shape shapeB1(shapeB.begin(), shapeB.begin() + (rankB - 2));
    Shape ret = infer_broadcast(shapeA1, shapeB1);
    if (ret.empty()) {
        b = 1;
    } else {
        b = std::accumulate(ret.begin(), ret.end(), 1, std::multiplies<int>());
    }
    IT_ASSERT(*(shapeA.rbegin()) == *(shapeB.rbegin() + 1));
    m = *(shapeA.rbegin() + 1);
    n = *(shapeB.rbegin());
    k = *(shapeA.rbegin());
    ret.emplace_back(m);
    ret.emplace_back(n);
    return {{ret}};
}

vector<DataType>
MatmulIntegerObj::inferDataType(const TensorVec &inputs) const {
    for (auto &input : inputs) {
        IT_ASSERT(input->getDType() == DataType::Int8 ||
                  input->getDType() == DataType::UInt8);
    }
    if (inputs.size() >= 3) {
        IT_ASSERT(inputs[0]->getDType() == inputs[2]->getDType());
    }
    if (inputs.size() == 4) {
        IT_ASSERT(inputs[1]->getDType() == inputs[3]->getDType());
    }
    return vector(numOutputs(), DataType::Int32);
}

vector<int> MatmulIntegerObj::getWorkloadVector() const {
    return {type.underlying(), b, m, n, k};
}

vector<int> MatmulIntegerObj::getOpAttrVector() const {
    return {type.underlying()};
}

} // namespace infini
