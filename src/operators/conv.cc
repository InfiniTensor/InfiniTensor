#include "operators/conv.h"

namespace infini {

ConvBaseObj::ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                         const vector<int> &pads, const vector<int> &strides,
                         const vector<int> &dilations,
                         const Tensor &inputInConvFWD,
                         const Tensor &weightInConvFWD, ActType act)
    : OperatorObj(opType, inputs, {output}), pads(pads), strides(strides),
      dilations(dilations), padding(PaddingMode::Other), act(act) {}
ConvBaseObj::ConvBaseObj(OpType opType, TensorVec inputs, Tensor &output,
                         PaddingMode mode, const vector<int> &strides,
                         const vector<int> &dilations,
                         const Tensor &inputInConvFWD,
                         const Tensor &weightInConvFWD, ActType act)
    : OperatorObj(opType, inputs, {output}), strides(strides),
      dilations(dilations), padding(mode), act(act) {
    IT_ASSERT(mode != PaddingMode::Other);
}

string ConvBaseObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "p=" << vecToString(pads) << ",";
    os << "s=" << vecToString(strides) << ",";
    os << "d=" << vecToString(dilations) << ",";
    // os << "act=" << enum_to_underlying(act) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> ConvBaseObj::getWorkloadVector() const {
    return {type.underlying(), n, c, h, w, f, r, s,
            pads[0],           pads[1], strides[0], strides[1],
            dilations[0],      dilations[1]};
}

vector<int> ConvBaseObj::getOpAttrVector() const {
    // IT_TODO_HALT(); // should padding mode / ph+pw be in attrs?
    return {type.underlying(), c, f, r, s,
            pads[0],           pads[1], strides[0], strides[1],
            dilations[0],      dilations[1]};
}

void ConvObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], c = input->getDims()[1], h = input->getDims()[2],
    w = input->getDims()[3], f = weight->getDims()[0], r = weight->getDims()[2],
    s = weight->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / strides[0];
        int ow = w / strides[1];
        int ph = (h - oh * strides[0] + (r - strides[0]) * dilations[0]) / 2;
        int pw = (w - ow * strides[1] + (s - strides[1]) * dilations[1]) / 2;
        pads = {ph, pw};
    } else if (mode == PaddingMode::Valid) {
        pads = {0, 0};
    }
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 int ph, int pw, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : ConvBaseObj(OpType::Conv, {input, weight}, output, {ph, pw}, {sh, sw},
                  {dh, dw}, input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 PaddingMode mode, int sh, int sw, int dh, int dw, Tensor bias,
                 ActType act)
    : ConvBaseObj(OpType::Conv, {input, weight}, output, mode, {sh, sw},
                  {dh, dw}, input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConvObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0], &weight = inputs[1];
    n = input->getDims()[0];
    c = input->getDims()[1];
    h = input->getDims()[2];
    w = input->getDims()[3];
    f = weight->getDims()[0];
    r = weight->getDims()[2];
    s = weight->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // For NCHW+FCRS layout, C of input is divisable by C of weight
    IT_ASSERT(input->getDims()[1] % weight->getDims()[1] == 0);
    // Set padding size
    if (padding == PaddingMode::Other) {
        oh = (h - (r - strides[0]) * dilations[0] + pads[0] * 2) / strides[0];
        ow = (w - (s - strides[1]) * dilations[1] + pads[1] * 2) / strides[1];
    } else if (padding == PaddingMode::Same) {
        oh = h / strides[0];
        ow = w / strides[1];
        // ph = (h - oh * sh + (r - sh) * dh) / 2;
        // pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (padding == PaddingMode::Valid) {
        int ph = 0;
        int pw = 0;
        oh = (h - (r - strides[0]) * dilations[0] + ph * 2) / strides[0];
        ow = (w - (s - strides[1]) * dilations[1] + pw * 2) / strides[1];
    }
    return {{{on, oc, oh, ow}}};
}

void Conv3dObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0];
    c = input->getDims()[1];
    d = input->getDims()[2];
    h = input->getDims()[3];
    w = input->getDims()[4];
    f = weight->getDims()[0];
    q = weight->getDims()[2];
    r = weight->getDims()[3];
    s = weight->getDims()[4];
    if (mode == PaddingMode::Same) {
        int od = d / strides[0];
        int oh = h / strides[1];
        int ow = w / strides[2];
        int pd = (d - od * strides[0] + (q - strides[0]) * dilations[0]) / 2;
        int ph = (h - oh * strides[1] + (r - strides[1]) * dilations[1]) / 2;
        int pw = (w - ow * strides[2] + (s - strides[2]) * dilations[2]) / 2;
        pads = {pd, ph, pw};
    } else if (mode == PaddingMode::Valid) {
        pads = {0, 0, 0};
    }
}

Conv3dObj::Conv3dObj(GraphObj *graph, Tensor input, Tensor weight,
                     Tensor output, int pd, int ph, int pw, int sd, int sh,
                     int sw, int dd, int dh, int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv3d, {input, weight}, output, {pd, ph, pw},
                  {sd, sh, sw}, {dd, dh, dw}, input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

Conv3dObj::Conv3dObj(GraphObj *graph, Tensor input, Tensor weight,
                     Tensor output, PaddingMode mode, int sd, int sh, int sw,
                     int dd, int dh, int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv3d, {input, weight}, output, mode, {sd, sh, sw},
                  {dd, dh, dw}, input, weight, act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

string Conv3dObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
    }
    os << "p=" << vecToString(pads) << ",";
    os << "s=" << vecToString(strides) << ",";
    os << "d=" << vecToString(dilations) << ",";
    // os << "act=" << enum_to_underlying(act) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<Shape>> Conv3dObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];
    const auto &weight = inputs[1];
    n = input->getDims()[0];
    c = input->getDims()[1];
    d = input->getDims()[2];
    h = input->getDims()[3];
    w = input->getDims()[4];
    f = weight->getDims()[0];
    q = weight->getDims()[2];
    r = weight->getDims()[3];
    s = weight->getDims()[4];
    int on = n;
    int oc = f;
    int od = 0;
    int oh = 0;
    int ow = 0;
    // For NCDHW+FCQRS layout, C of input is divisable by C of weight.
    IT_ASSERT(input->getDims()[1] % weight->getDims()[1] == 0);
    // Set padding size.
    if (padding == PaddingMode::Other) {
        od = (d - (q - strides[0]) * dilations[0] + pads[0] * 2) / strides[0];
        oh = (h - (r - strides[1]) * dilations[1] + pads[1] * 2) / strides[1];
        ow = (w - (s - strides[2]) * dilations[2] + pads[2] * 2) / strides[2];
    } else if (padding == PaddingMode::Same) {
        od = d / strides[0];
        oh = h / strides[1];
        ow = w / strides[2];
    } else if (padding == PaddingMode::Valid) {
        int pd = 0;
        int ph = 0;
        int pw = 0;
        od = (d - (q - strides[0]) * dilations[0] + pd * 2) / strides[0];
        oh = (h - (r - strides[1]) * dilations[1] + ph * 2) / strides[1];
        ow = (w - (s - strides[2]) * dilations[2] + pw * 2) / strides[2];
    }
    return {{{on, oc, od, oh, ow}}};
}

ConvTransposed2dObj::ConvTransposed2dObj(GraphObj *graph, Tensor input,
                                         Tensor weight, Tensor output, int ph,
                                         int pw, int sh, int sw, int dh, int dw,
                                         int oph, int opw, int group,
                                         Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, {ph, pw},
                  {sh, sw}, {dh, dw}, output, weight, act),
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
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, mode,
                  {sh, sw}, {dh, dw}, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvTransposed2dObj::inferShape(const TensorVec &inputs) {
    const Tensor &input = inputs[0], &weight = inputs[1];
    n = input->getDims()[0];
    f = input->getDims()[1];
    h = input->getDims()[2];
    w = input->getDims()[3];
    c = weight->getDims()[1];
    r = weight->getDims()[2];
    s = weight->getDims()[3];
    IT_ASSERT(f == weight->getDims()[0]);

    int on = n, oc = c * group;
    int oh = 0, ow = 0;
    oh = (h - 1) * strides[0] - 2 * pads[0] + dilations[0] * (r - 1) + oph + 1;
    ow = (w - 1) * strides[1] - 2 * pads[1] + dilations[1] * (s - 1) + opw + 1;
    return {{{on, oc, oh, ow}}};
}

void ConvTransposed2dObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], f = input->getDims()[1], h = input->getDims()[2],
    w = input->getDims()[3], c = weight->getDims()[1], r = weight->getDims()[2],
    s = weight->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / strides[0];
        int ow = w / strides[1];
        int ph =
            (h - oh * strides[0] + (r - strides[0]) * dilations[0]) / 2;
        int pw =
            (w - ow * strides[1] + (s - strides[1]) * dilations[1]) / 2;
        pads = {ph, pw};
    } else if (mode == PaddingMode::Valid) {
        pads = {0, 0};
    }
}

void ConvBackwardFilterObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &inputX = inputs[0];
    const Tensor &diffY = inputs[1];
    n = inputX->getDims()[0], c = inputX->getDims()[1],
    h = inputX->getDims()[2], w = inputX->getDims()[3], f = diffY->getDims()[0],
    r = diffY->getDims()[2], s = diffY->getDims()[3];
    if (mode == PaddingMode::Same) {
        int oh = h / strides[0];
        int ow = w / strides[1];
        int ph =
            (h - oh * strides[0] + (r - strides[0]) * dilations[0]) / 2;
        int pw =
            (w - ow * strides[1] + (s - strides[1]) * dilations[1]) / 2;
        pads = {ph, pw};
    } else if (mode == PaddingMode::Valid) {
        pads = {0, 0};
    }
}

ConvBackwardFilterObj::ConvBackwardFilterObj(GraphObj *graph, Tensor inputX,
                                             Tensor diffY, Tensor diffW, int ph,
                                             int pw, int sh, int sw, int dh,
                                             int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv, {inputX, diffY}, diffW, {ph, pw}, {sh, sw},
                  {dh, dw}, inputX, diffY),
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
    : ConvBaseObj(OpType::Conv, {inputX, diffY}, diffW, mode, {sh, sw},
                  {dh, dw}, inputX, diffY),
      act(act) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvBackwardFilterObj::inferShape(const TensorVec &inputs) {
    const auto &inputX = inputs[0], &diffY = inputs[1];
    n = inputX->getDims()[0];
    c = inputX->getDims()[1];
    h = inputX->getDims()[2];
    w = inputX->getDims()[3];
    f = diffY->getDims()[0];
    r = diffY->getDims()[2];
    s = diffY->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // For NCHW+FCRS layout, C of input is divisable by C of weight
    IT_ASSERT(inputX->getDims()[1] % diffY->getDims()[1] == 0);
    // Set padding size
    if (padding == PaddingMode::Other) {
        oh = (h - (r - strides[0]) * dilations[0] + pads[0] * 2) / strides[0];
        ow = (w - (s - strides[1]) * dilations[1] + pads[1] * 2) / strides[1];
    } else if (padding == PaddingMode::Same) {
        oh = h / strides[0];
        ow = w / strides[1];
        // ph = (h - oh * sh + (r - sh) * dh) / 2;
        // pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (padding == PaddingMode::Valid) {
        int ph = 0;
        int pw = 0;
        oh = (h - (r - strides[0]) * dilations[0] + ph * 2) / strides[0];
        ow = (w - (s - strides[1]) * dilations[1] + pw * 2) / strides[1];
    }
    return {{{on, oc, oh, ow}}};
}

ConvTransposed2dNHWCObj::ConvTransposed2dNHWCObj(GraphObj *graph, Tensor input,
                                                 Tensor weight, Tensor output,
                                                 int ph, int pw, int sh, int sw,
                                                 int dh, int dw, int oph,
                                                 int opw, int group,
                                                 Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTransNHWC, {input, weight}, output, {ph, pw},
                  {sh, sw}, {dh, dw}, output, weight, act),
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
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, mode,
                  {sh, sw}, {dh, dw}, output, weight, act),
      oph(oph), opw(opw), group(group) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>>
ConvTransposed2dNHWCObj::inferShape(const TensorVec &inputs) {
    const Tensor &input = inputs[0], &weight = inputs[1];
    n = input->getDims()[0];
    f = input->getDims()[3];
    h = input->getDims()[1];
    w = input->getDims()[2];
    c = weight->getDims()[3];
    r = weight->getDims()[1];
    s = weight->getDims()[2];
    IT_ASSERT(f == weight->getDims()[0]);

    int on = n, oc = c * group;
    int oh = 0, ow = 0;
    oh = (h - 1) * strides[0] - 2 * pads[0] + dilations[0] * (r - 1) + oph + 1;
    ow = (w - 1) * strides[1] - 2 * pads[1] + dilations[1] * (s - 1) + opw + 1;
    return {{{on, oh, ow, oc}}};
}

void ConvTransposed2dNHWCObj::setAuxilaryAttributes(PaddingMode mode) {
    const Tensor &input = inputs[0];
    const Tensor &weight = inputs[1];
    n = input->getDims()[0], f = input->getDims()[3], h = input->getDims()[1],
    w = input->getDims()[2], c = weight->getDims()[3], r = weight->getDims()[1],
    s = weight->getDims()[2];
    if (mode == PaddingMode::Same) {
        int oh = h / strides[0];
        int ow = w / strides[1];
        int ph =
            (h - oh * strides[0] + (r - strides[0]) * dilations[0]) / 2;
        int pw =
            (w - ow * strides[1] + (s - strides[1]) * dilations[1]) / 2;
        pads = {ph, pw};
    } else if (mode == PaddingMode::Valid) {
        pads = {0, 0};
    }
}

} // namespace infini
