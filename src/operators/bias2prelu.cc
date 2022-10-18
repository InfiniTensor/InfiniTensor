#include "operators/bias2prelu.h"

namespace infini {

BiasPReLU::BiasPReLU(GraphObj *graph, Tensor input, Tensor bias, Tensor output,
                     bool PReLU_, float paramPReLU_)
    : OperatorObj(OpType::BiasPReLU, {input, bias}, {output}), PReLU(PReLU_),
      paramPReLU(paramPReLU_) {
    auto dims = input->getDims();
    n = dims[0], h = dims[1], w = dims[2], c = dims[3];

    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> BiasPReLU::inferShape(const TensorVec &inputs) const {
    const Tensor &input = inputs[0];
    const Tensor &bias = inputs[1];

    auto dims = input->getDims();
    int n = dims[0], h = dims[1], w = dims[2], c = dims[3];
    int bc = bias->getDims()[0];

    if (bc != c)
        return {};

    return {{{n, h, w, c}}};
}

vector<int> BiasPReLU::getWorkloadVector() const {
    return {enum_to_underlying(type), n, h, w, c};
}

vector<int> BiasPReLU::getOpAttrVector() const {
    return {enum_to_underlying(type)};
}

} // namespace infini