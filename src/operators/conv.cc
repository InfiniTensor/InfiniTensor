#include "operators/conv.h"

namespace infini {

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 int ph, int pw, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : OperatorObj(OpType::Conv, {input, weight, bias}, {output}), ph(ph),
      pw(pw), sh(sh), sw(sw), dh(dh), dw(dw), act(act),
      padding(PaddingMode::Other) {
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 PaddingMode mode, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : OperatorObj(OpType::Conv, {input, weight, bias}, {output}), ph(-1),
      pw(-1), sh(sh), sw(sw), dh(dh), dw(dw), act(act), padding(mode) {
    IT_ASSERT(mode != PaddingMode::Other);
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

string ConvObj::toString() const {
    std::ostringstream os;
    os << "Conv[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "act=" << enum_to_underlying(act) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<Shape>> ConvObj::inferShape(const TensorVec &inputs) const {
    const auto &input = inputs[0], &weight = inputs[1];
    auto n = input->getDims()[0];
    auto h = input->getDims()[2];
    auto w = input->getDims()[3];
    auto f = weight->getDims()[0];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // For NCHW+FCRS layout, C of input is divisable by C of weight
    if (input->getDims()[1] % weight->getDims()[1] != 0)
        return {};
    // Set padding size
    if (padding == PaddingMode::Other) {
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    } else if (padding == PaddingMode::Same) {
        oh = h / sh;
        ow = w / sw;
        // ph = (h - oh * sh + (r - sh) * dh) / 2;
        // pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (padding == PaddingMode::Valid) {
        int ph = 0;
        int pw = 0;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    }
    return {{{on, oc, oh, ow}}};
}

vector<int> ConvObj::getWorkloadVector() const {
    return {
        enum_to_underlying(type), n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw,
        enum_to_underlying(act)};
}

vector<int> ConvObj::getOpAttrVector() const {
    IT_TODO_HALT(); // should padding mode / ph+pw be in attrs?
    return {enum_to_underlying(type), c, f, r, s, ph, pw, sh, sw, dh, dw,
            enum_to_underlying(act)};
}

void ConvObj::setAuxilaryAttributes(PaddingMode mode) {
    n = inputs[0]->getDims()[0], c = inputs[0]->getDims()[1],
    h = inputs[0]->getDims()[2], w = inputs[0]->getDims()[3],
    f = inputs[1]->getDims()[0], r = inputs[1]->getDims()[2],
    s = inputs[1]->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

} // namespace infini