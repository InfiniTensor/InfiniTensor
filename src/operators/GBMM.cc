#include "operators/GBMM.h"

namespace infini {

GBMMObj::GBMMObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, int dilation,
                 [[maybe_unused]] Tensor bias, ActType act)
    : OperatorObj(OpType::GBMM, {A, B}, {C}), dilation(dilation), act(act),
      b(A->getDims()[0]), m(A->getDims()[1]), w((A->getDims()[2] - 1) / 2),
      n(B->getDims()[2]) {
    IT_ASSERT(checkValid(graph));
}

string GBMMObj::toString() const {
    std::ostringstream os;
    os << "GBMM(["
       << ",act=" << (int)act << "],A=" << inputs[0]->getGuid()
       << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
       << ", TTbmwnd: " << this->getB() << ", " << this->getM() << ", "
       << this->getW() << ", " << this->getN() << ", " << this->getDilation()
       << ")";
    return os.str();
}

optional<vector<Shape>> GBMMObj::inferShape(const TensorVec &inputs) const {
    auto A = inputs[0], B = inputs[1];

    if (!(A->getDims().size() == 3 && B->getDims().size() == 3))
        return {};
    if (!(A->getDims()[0] == B->getDims()[0]))
        return {};
    if (!(A->getDims()[1] == B->getDims()[1]))
        return {};
    if (A->getDims()[2] % 2 == 0)
        return {};
    int b(A->getDims()[0]), m(A->getDims()[1]), k(B->getDims()[2]);
    return {{{b, m, k}}};
}

vector<int> GBMMObj::getWorkloadVector() const {
    return {enum_to_underlying(type), b, m, w, n, dilation,
            enum_to_underlying(act)};
}

vector<int> GBMMObj::getOpAttrVector() const {
    return {enum_to_underlying(type), dilation, enum_to_underlying(act)};
}
} // namespace infini
