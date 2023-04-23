#include "operators/conv.h"

namespace infini {

ConvBaseObj::ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                         int ph, int pw, int sh, int sw, int dh, int dw,
                         const Tensor &inputInConvFWD,
                         const Tensor &weightInConvFWD, ActType act)
    : OperatorObj(opType, inputs, {output}), ph(ph), pw(pw), sh(sh), sw(sw),
      dh(dh), dw(dw), padding(PaddingMode::Other), act(act) {}
ConvBaseObj::ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                         PaddingMode mode, int sh, int sw, int dh, int dw,
                         const Tensor &inputInConvFWD,
                         const Tensor &weightInConvFWD, ActType act)
    : OperatorObj(opType, inputs, {output}), ph(-1), pw(-1), sh(sh), sw(sw),
      dh(dh), dw(dw), padding(mode), act(act) {
    IT_ASSERT(mode != PaddingMode::Other);
}

string ConvBaseObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(getOpType()) << "[" << getGuid() << "]";
    os << "(";
    for (auto &input : inputs) {
        os << vecToString(input->getDims()) << ",";
    }
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    // os << "act=" << enum_to_underlying(act) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ConvBaseObj::getWorkloadVector() const {
    return {
        enum_to_underlying(type), n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw};
}

vector<int> ConvBaseObj::getOpAttrVector() const {
    // IT_TODO_HALT(); // should padding mode / ph+pw be in attrs?
    return {enum_to_underlying(type), c, f, r, s, ph, pw, sh, sw, dh, dw};
}

void ConvObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], c = input->getDims()[1], h = input->getDims()[2],
    w = input->getDims()[3], f = weight->getDims()[0], r = weight->getDims()[2],
    s = weight->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 int ph, int pw, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : ConvBaseObj(OpType::Conv, {input, weight}, output, ph, pw, sh, sw, dh, dw,
                  input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 PaddingMode mode, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : ConvBaseObj(OpType::Conv, {input, weight}, output, mode, sh, sw, dh, dw,
                  input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
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

void ConvNHWCObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], c = input->getDims()[3], h = input->getDims()[1],
    w = input->getDims()[2], f = weight->getDims()[0], r = weight->getDims()[1],
    s = weight->getDims()[2];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

ConvNHWCObj::ConvNHWCObj(GraphObj *graph, Tensor input, Tensor weight,
                         Tensor output, int ph, int pw, int sh, int sw, int dh,
                         int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvNHWC, {input, weight}, output, ph, pw, sh, sw, dh,
                  dw, input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvNHWCObj::ConvNHWCObj(GraphObj *graph, Tensor input, Tensor weight,
                         Tensor output, PaddingMode mode, int sh, int sw,
                         int dh, int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvNHWC, {input, weight}, output, mode, sh, sw, dh,
                  dw, input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConvNHWCObj::inferShape(const TensorVec &inputs) const {
    const auto &input = inputs[0], &weight = inputs[1];
    auto n = input->getDims()[0];
    auto h = input->getDims()[1];
    auto w = input->getDims()[2];
    auto f = weight->getDims()[0];
    auto r = weight->getDims()[1];
    auto s = weight->getDims()[2];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // For NCHW+FCRS layout, C of input is divisable by C of weight
    if (input->getDims()[3] % weight->getDims()[3] != 0)
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
    return {{{on, oh, ow, oc}}};
}

ConvTransposed2dObj::ConvTransposed2dObj(GraphObj *graph, Tensor input,
                                         Tensor weight, Tensor output, int ph,
                                         int pw, int sh, int sw, int dh, int dw,
                                         int oph, int opw, int group,
                                         Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTrans, {input, weight}, output, ph, pw, sh, sw,
                  dh, dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvTransposed2dObj::ConvTransposed2dObj(GraphObj *graph, Tensor input,
                                         Tensor weight, Tensor output,
                                         PaddingMode mode, int sh, int sw,
                                         int dh, int dw, int oph, int opw,
                                         int group, Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTrans, {input, weight}, output, mode, sh, sw, dh,
                  dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvTransposed2dObj::inferShape(const TensorVec &inputs) const {
    const Tensor &input = inputs[0], &weight = inputs[1];
    auto n = input->getDims()[0];
    auto f = input->getDims()[1];
    auto h = input->getDims()[2];
    auto w = input->getDims()[3];
    auto c = weight->getDims()[1];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    if (f != weight->getDims()[0])
        return {};

    int on = n, oc = c * group;
    int oh = 0, ow = 0;
    oh = (h - 1) * sh - 2 * ph + dh * (r - 1) + oph + 1;
    ow = (w - 1) * sw - 2 * pw + dw * (s - 1) + opw + 1;
    return {{{on, oc, oh, ow}}};
}

void ConvTransposed2dObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], f = input->getDims()[1], h = input->getDims()[2],
    w = input->getDims()[3], c = weight->getDims()[1], r = weight->getDims()[2],
    s = weight->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

void ConvBackwardFilterObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &inputX = inputs[0];
    const Tensor &diffY = inputs[1];
    n = inputX->getDims()[0], c = inputX->getDims()[1],
    h = inputX->getDims()[2], w = inputX->getDims()[3], f = diffY->getDims()[0],
    r = diffY->getDims()[2], s = diffY->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / sh;
        int ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        ph = pw = 0;
    }
}

ConvBackwardFilterObj::ConvBackwardFilterObj(GraphObj *graph, Tensor inputX,
                                             Tensor diffY, Tensor diffW, int ph,
                                             int pw, int sh, int sw, int dh,
                                             int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv, {inputX, diffY}, diffW, ph, pw, sh, sw, dh, dw,
                  inputX, diffY),
      act(act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvBackwardFilterObj::ConvBackwardFilterObj(GraphObj *graph, Tensor inputX,
                                             Tensor diffY, Tensor diffW,
                                             PaddingMode mode, int sh, int sw,
                                             int dh, int dw, Tensor bias,
                                             ActType act)
    : ConvBaseObj(OpType::Conv, {inputX, diffY}, diffW, mode, sh, sw, dh, dw,
                  inputX, diffY),
      act(act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvBackwardFilterObj::inferShape(const TensorVec &inputs) const {
    const auto &inputX = inputs[0], &diffY = inputs[1];
    auto n = inputX->getDims()[0];
    auto h = inputX->getDims()[2];
    auto w = inputX->getDims()[3];
    auto f = diffY->getDims()[0];
    auto r = diffY->getDims()[2];
    auto s = diffY->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // For NCHW+FCRS layout, C of input is divisable by C of weight
    if (inputX->getDims()[1] % diffY->getDims()[1] != 0)
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

ConvTransposed2dNHWCObj::ConvTransposed2dNHWCObj(GraphObj *graph, Tensor input,
                                                 Tensor weight, Tensor output,
                                                 int ph, int pw, int sh, int sw,
                                                 int dh, int dw, int oph,
                                                 int opw, int group,
                                                 Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTransNHWC, {input, weight}, output, ph, pw, sh,
                  sw, dh, dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvTransposed2dNHWCObj::ConvTransposed2dNHWCObj(GraphObj *graph, Tensor input,
                                                 Tensor weight, Tensor output,
                                                 PaddingMode mode, int sh,
                                                 int sw, int dh, int dw,
                                                 int oph, int opw, int group,
                                                 Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTrans, {input, weight}, output, mode, sh, sw, dh,
                  dw, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvTransposed2dNHWCObj::inferShape(const TensorVec &inputs) const {
    const Tensor &input = inputs[0], &weight = inputs[1];
    auto n = input->getDims()[0];
    auto f = input->getDims()[3];
    auto h = input->getDims()[1];
    auto w = input->getDims()[2];
    auto c = weight->getDims()[3];
    auto r = weight->getDims()[1];
    auto s = weight->getDims()[2];
    if (f != weight->getDims()[0])
        return {};

    int on = n, oc = c * group;
    int oh = 0, ow = 0;
    oh = (h - 1) * sh - 2 * ph + dh * (r - 1) + oph + 1;
    ow = (w - 1) * sw - 2 * pw + dw * (s - 1) + opw + 1;
    return {{{on, oh, ow, oc}}};
}

void ConvTransposed2dNHWCObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], f = input->getDims()[3], h = input->getDims()[1],
    w = input->getDims()[2], c = weight->getDims()[3], r = weight->getDims()[1],
    s = weight->getDims()[2];
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
