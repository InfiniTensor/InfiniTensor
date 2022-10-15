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

optional<vector<Shape>> G2BMMObj::inferShape(const TensorVec &inputs) const {
    auto A = inputs[0], B = inputs[1];

    if (!(A->getDims().size() == 3 && B->getDims().size() == 3))
        return {};
    if (!(A->getDims()[0] == B->getDims()[0]))
        return {};
    if (!(A->getDims()[1] == B->getDims()[1]))
        return {};
    if (!(A->getDims()[2] == B->getDims()[2]))
        return {};
    if (width < 0)
        return {};
    int b(A->getDims()[0]), m(A->getDims()[1]), n(2 * width + 1);
    return {{{b, m, n}}};
}

vector<int> G2BMMObj::getWorkloadVector() const {
    return {enum_to_underlying(type), b, m, k, width, dilation,
            enum_to_underlying(act)};
}

vector<int> G2BMMObj::getOpAttrVector() const {
    return {enum_to_underlying(type), width, dilation, enum_to_underlying(act)};
}

} // namespace infini
