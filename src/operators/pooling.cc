#include "operators/pooling.h"

namespace infini {

PoolingObj::PoolingObj(GraphObj *graph, OpType optype, Tensor input,
                       Tensor output, int kh, int kw, int dh, int dw, int ph,
                       int pw, int sh, int sw, int ceilMode)
    : OperatorObj(optype, {input}, {output}), kh(kh), kw(kw), dh(dh), dw(dw),
      ph(ph), pw(pw), sh(sh), sw(sw), ceilMode(ceilMode),
      n(input->getDims().at(0)), c(input->getDims().at(1)),
      h(input->getRank() == 3 ? 1 : input->getDims().at(2)),
      w(input->getRank() == 3 ? input->getDims().at(2)
                              : input->getDims().at(3)) {
    IT_ASSERT(checkValid(graph));
}

void PoolingObj::initInfiniOp(const Runtime context) {
    auto x_dim = inputs[0]->getDims();
    auto y_dim = outputs[0]->getDims();

    uint64_t kernel_shape[2] = {(uint64_t)kh, (uint64_t)kw};
    uint64_t pads[2] = {(uint64_t)ph, (uint64_t)pw};
    int64_t strides[2] = {(int64_t)sh, (int64_t)sw};

    auto x_shape = toInfiniopShape(x_dim);
    auto y_shape = toInfiniopShape(y_dim);

    // create tensor descriptor
    infiniopTensorDescriptor_t x_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &x_tensor, x_dim.size(), x_shape.data(), nullptr,
        toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    infiniopTensorDescriptor_t y_tensor;
    CHECK_ERROR(infiniopCreateTensorDescriptor(
        &y_tensor, y_dim.size(), y_shape.data(), nullptr,
        toInfiniopDataLayout(outputs[0]->getDType().getIndex())));

    // create op descriptor
    if (type == OpType::MaxPool) {
        CHECK_ERROR(infiniopCreateMaxPoolDescriptor(
            context->opHandle(), (infiniopMaxPoolDescriptor_t *)&opDesc,
            y_tensor, x_tensor, kernel_shape, pads, strides, 2));
    } else if (type == OpType::AveragePool) {
        CHECK_ERROR(infiniopCreateAvgPoolDescriptor(
            context->opHandle(), (infiniopAvgPoolDescriptor_t *)&opDesc,
            y_tensor, x_tensor, kernel_shape, pads, strides, 2));
    } else {
        opDesc = nullptr;
    }

    // destroy tensor descriptor
    CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
}

optional<vector<Shape>> PoolingObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];
    int oh, ow;
    if (ceilMode) {
        oh = ceil(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = ceil(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    } else {
        oh = floor(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = floor(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    }

    auto ret = input->getDims();
    if (input->getRank() == 4) {
        ret[input->getRank() - 2] = oh;
    }
    ret[input->getRank() - 1] = ow;
    return {{ret}};
}

std::string PoolingObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << "k=[" << kh << "," << kw << "],";
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "ceil mode=" << ceilMode << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PoolingObj::getWorkloadVector() const {
    return {type.underlying(), n, c, h, w, kh, kw, ph, pw, sh, sw, dh, dw,
            ceilMode};
}

vector<int> PoolingObj::getOpAttrVector() const {
    return {type.underlying(), kh, kw, ph, pw, sh, sw, dh, dw, ceilMode};
}

}; // namespace infini
