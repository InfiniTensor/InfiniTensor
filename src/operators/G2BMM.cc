#include "operators/G2BMM.h"

namespace infini {

G2BMMObj::G2BMMObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, int width,
                   int dilation, [[maybe_unused]] Tensor bias, ActType act)
    : OperatorObj(OpType::G2BMM, {A, B}, {C}), width(width), dilation(dilation),
      act(act), b(A->getDims()[0]), m(A->getDims()[1]), k(A->getDims()[2]) {
    IT_ASSERT(checkValid(graph));
}

string G2BMMObj::toString() const {
    std::ostringstream os;
    os << "G2BMM(["
       << "width=" << width << ",act=" << enum_to_underlying(act)
       << "],A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
       << ",C=" << outputs[0]->getGuid() << ", TTbmnkd: " << this->getB()
       << ", " << this->getM() << ", " << this->getWidth() << ", "
       << inputs[1]->getDims()[2] << ", " << this->getDilation() << ")";
    return os.str();
}

optional<vector<Shape>> G2BMMObj::inferShape(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    b = A->getDims()[0];
    m = A->getDims()[1];
    k = A->getDims()[2];

    IT_ASSERT(A->getRank() == 3 && B->getRank() == 3);
    IT_ASSERT(A->getDims()[0] == B->getDims()[0]);
    IT_ASSERT(A->getDims()[1] == B->getDims()[1]);
    IT_ASSERT(A->getDims()[2] == B->getDims()[2]);
    IT_ASSERT(width >= 0);
    int n(2 * width + 1);
    return {{{b, m, n}}};
}

vector<int> G2BMMObj::getWorkloadVector() const {
    return {type.underlying(),      b, m, k, width, dilation,
            enum_to_underlying(act)};
}

vector<int> G2BMMObj::getOpAttrVector() const {
    return {type.underlying(), width, dilation, enum_to_underlying(act)};
}

} // namespace infini
