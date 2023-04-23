#include "operators/conv2dreduce.h"

namespace infini {

Conv2dReduceBase::Conv2dReduceBase(OpType opType, Tensor input, Tensor bias_,
                                   Tensor output, bool PReLU_, float paramReLU_,
                                   int ph_, int pw_, int sh_, int sw_, int dh_,
                                   int dw_)
    : OperatorObj(opType, {input}, {output}), bias(bias_), ph(ph_), pw(pw_),
      sh(sh_), sw(sw_), dh(dh_), dw(dw_), PReLU(PReLU_), paramReLU(paramReLU_) {
    // expect input shape is (n, h, w, f, r, s)
    auto inputShape = input->getDims();
    IT_ASSERT(inputShape.size() == 6);
    n = inputShape[0];
    h = inputShape[1];
    w = inputShape[2];
    f = inputShape[3];
    r = inputShape[4];
    s = inputShape[5];

    if (bias) {
        auto biasShape = bias->getDims();
        IT_ASSERT(biasShape.size() == 1);
        IT_ASSERT(biasShape[0] == f);
    }
}

std::string Conv2dReduceBase::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(getOpType()) << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    } else {
        os << vecToString(inputs[0]->getDims()) << ",";
    }
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "PReLU=" << (PReLU ? "true" : "false") << ",";
    // os << "act=" << enum_to_underlying(act) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    if (bias != nullptr) {
        os << "bias=" << bias->getGuid() << ",";
    }
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

std::vector<int> Conv2dReduceBase::getWorkloadVector() const {
    return {enum_to_underlying(type), n, h, w, f, r, s, ph, pw, sh, sw, dh, dw};
}

std::vector<int> Conv2dReduceBase::getOpAttrVector() const {
    return {enum_to_underlying(type), ph, pw, sh, sw, dh, dw};
}

Conv2dReduce::Conv2dReduce(GraphObj *graph, Tensor input, Tensor bias,
                           Tensor output, bool PReLU_, float paramReLU_,
                           int ph_, int pw_, int sh_, int sw_, int dh_, int dw_)
    : Conv2dReduceBase(OpType::Conv2dReduce, input, bias, output, PReLU_,
                       paramReLU_, ph_, pw_, sh_, sw_, dh_, dw_) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
Conv2dReduce::inferShape(const TensorVec &inputs) const {
    // const auto &input = inputs[0], &bias = inputs[1];
    int on = n, of = f;
    int oh = (h + ph * 2 - dh * (r - 1) - 1) / sh + 1;
    int ow = (w + pw * 2 - dw * (s - 1) - 1) / sw + 1;

    return {{{on, oh, ow, of}}};
}

Conv2dReduceTranspose::Conv2dReduceTranspose(GraphObj *graph, Tensor input,
                                             Tensor bias, Tensor output,
                                             bool PReLU_, float paramReLU_,
                                             int ph_, int pw_, int sh_, int sw_,
                                             int dh_, int dw_)
    : Conv2dReduceBase(OpType::Conv2dReduceTranspose, input, bias, output,
                       PReLU_, paramReLU_, ph_, pw_, sh_, sw_, dh_, dw_) {
    IT_ASSERT(dh_ == 1);
    IT_ASSERT(dw_ == 1);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
Conv2dReduceTranspose::inferShape(const TensorVec &inputs) const {
    // const auto &input = inputs[0], &bias = inputs[1];
    int on = n, of = f;
    int oh = (h - 1) * sh - 2 * ph + dh * (r - 1) + 1;
    int ow = (w - 1) * sw - 2 * pw + dw * (s - 1) + 1;

    return {{{on, oh, ow, of}}};
}
} // namespace infini
