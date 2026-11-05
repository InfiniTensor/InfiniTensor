#include "operators/rms_norm.h"

#include <cmath>
#include <cstring>

namespace infini {
RMSNormObj::RMSNormObj(GraphObj *graph, Tensor input, Tensor weight,
                       Tensor output, float epsilon)
    : OperatorObj(OpType::RMSNorm, {input, weight}, {output}), epsilon(epsilon) {
    IT_ASSERT(std::isfinite(epsilon) && epsilon >= 0);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> RMSNormObj::inferShape(const TensorVec &inputs) {
    const auto A = inputs[0];
    auto input_dim = A->getDims();
    auto output_dim = input_dim;
    return {{output_dim}};
}

std::string RMSNormObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> RMSNormObj::getWorkloadVector() const {
    auto ret = getOpAttrVector();
    const Shape shape = outputs[0]->getDims();
    ret.insert(ret.end(), shape.begin(), shape.end());
    return ret;
}

vector<int> RMSNormObj::getOpAttrVector() const {
    static_assert(sizeof(int) == sizeof(float));
    int bits = 0;
    std::memcpy(&bits, &epsilon, sizeof(bits));
    return {type.underlying(), bits};
}

}; // namespace infini
