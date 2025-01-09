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
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << vecToString(inputs[0]->getDims()) << ",";
        os << vecToString(inputs[1]->getDims()) << ",";
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
    return {type.underlying(), n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw};
}

vector<int> ConvBaseObj::getOpAttrVector() const {
    // IT_TODO_HALT(); // should padding mode / ph+pw be in attrs?
    return {type.underlying(), c, f, r, s, ph, pw, sh, sw, dh, dw};
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
                 int ph, int pw, Tensor bias, int sh, int sw, int dh, int dw,
                 ActType act)
    : ConvBaseObj(OpType::Conv,
                  bias ? TensorVec{input, weight, bias}
                       : TensorVec{input, weight},
                  output, ph, pw, sh, sw, dh, dw, input, weight, act) {
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 Tensor bias, PaddingMode mode, int sh, int sw, int dh, int dw,
                 ActType act)
    : ConvBaseObj(OpType::Conv,
                  bias ? TensorVec{input, weight, bias}
                       : TensorVec{input, weight},
                  output, mode, sh, sw, dh, dw, input, weight, act) {
    setAuxilaryAttributes(mode);
    IT_ASSERT(checkValid(graph));
}

void ConvObj::initInfiniOp(const Runtime context) {
    // auto x_dim = inputs[0]->getDims();
    // auto w_dim = inputs[1]->getDims();
    // auto y_dim = outputs[0]->getDims();
    // uint64_t pads[2] = {(uint64_t)ph, (uint64_t)pw};
    // int64_t strides[2] = {(int64_t)sh, (int64_t)sw};
    // uint64_t dilations[2] = {(uint64_t)dh, (uint64_t)dw};

    // auto x_shape = toInfiniopShape(x_dim);
    // auto w_shape = toInfiniopShape(w_dim);
    // auto y_shape = toInfiniopShape(y_dim);
    // // create tensor descriptor
    // infiniopTensorDescriptor_t x_tensor;
    // CHECK_ERROR(infiniopCreateTensorDescriptor(
    //     &x_tensor, x_dim.size(), x_shape.data(), nullptr,
    //     toInfiniopDataLayout(inputs[0]->getDType().getIndex())));
    // infiniopTensorDescriptor_t w_tensor;
    // CHECK_ERROR(infiniopCreateTensorDescriptor(
    //     &w_tensor, w_dim.size(), w_shape.data(), nullptr,
    //     toInfiniopDataLayout(inputs[1]->getDType().getIndex())));
    // infiniopTensorDescriptor_t y_tensor;
    // CHECK_ERROR(infiniopCreateTensorDescriptor(
    //     &y_tensor, y_dim.size(), y_shape.data(), nullptr,
    //     toInfiniopDataLayout(outputs[0]->getDType().getIndex())));
    // if (inputs.size() == 2) {
    //     // create op descriptor
    //     CHECK_ERROR(infiniopCreateConvDescriptor(
    //         context->opHandle(), (infiniopConvDescriptor_t *)&opDesc,
    //         y_tensor, x_tensor, w_tensor, pads, strides, dilations, 2));
    // } else if (inputs.size() == 3) {
    //     auto b_dim = inputs[2]->getDims();
    //     auto b_shape = toInfiniopShape(b_dim);
    //     infiniopTensorDescriptor_t b_tensor;
    //     CHECK_ERROR(infiniopCreateTensorDescriptor(
    //         &b_tensor, b_dim.size(), b_shape.data(), nullptr,
    //         toInfiniopDataLayout(inputs[2]->getDType().getIndex())));
    //     CHECK_ERROR(infiniopCreateConvBiasActDescriptor(
    //         context->opHandle(), (infiniopConvBiasActDescriptor_t *)&opDesc,
    //         y_tensor, x_tensor, w_tensor, b_tensor, pads, strides, dilations,
    //         2, 0));
    //     CHECK_ERROR(infiniopDestroyTensorDescriptor(b_tensor));
    // } else {
    //     IT_ASSERT(false);
    // }

    // // destroy tensor descriptor and op descriptor
    // CHECK_ERROR(infiniopDestroyTensorDescriptor(y_tensor));
    // CHECK_ERROR(infiniopDestroyTensorDescriptor(x_tensor));
    // CHECK_ERROR(infiniopDestroyTensorDescriptor(w_tensor));
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
        int od = d / sd;
        int oh = h / sh;
        int ow = w / sw;
        pd = (d - od * sd + (q - sd) * dd) / 2;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (mode == PaddingMode::Valid) {
        pd = 0;
        ph = 0;
        pw = 0;
    }
}

Conv3dObj::Conv3dObj(GraphObj *graph, Tensor input, Tensor weight,
                     Tensor output, int pd, int ph, int pw, int sd, int sh,
                     int sw, int dd, int dh, int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv3d, {input, weight}, output, ph, pw, sh, sw, dh,
                  dw, input, weight, act),
      pd(pd), sd(sd), dd(dd) {
    if (bias)
        IT_TODO_HALT();
    setAuxilaryAttributes(PaddingMode::Other);
    IT_ASSERT(checkValid(graph));
}

Conv3dObj::Conv3dObj(GraphObj *graph, Tensor input, Tensor weight,
                     Tensor output, PaddingMode mode, int sd, int sh, int sw,
                     int dd, int dh, int dw, Tensor bias, ActType act)
    : ConvBaseObj(OpType::Conv3d, {input, weight}, output, mode, sh, sw, dh, dw,
                  input, weight, act),
      sd(sd), dd(dd) {
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
    os << "p=[" << pd << "," << ph << "," << pw << "],";
    os << "s=[" << sd << "," << sh << "," << sw << "],";
    os << "d=[" << dd << "," << dh << "," << dw << "],";
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
        od = (d - (q - sd) * dd + pd * 2) / sd;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    } else if (padding == PaddingMode::Same) {
        od = d / sd;
        oh = h / sh;
        ow = w / sw;
    } else if (padding == PaddingMode::Valid) {
        int pd = 0;
        int ph = 0;
        int pw = 0;
        od = (d - (q - sd) * dd + pd * 2) / sd;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    }
    return {{{on, oc, od, oh, ow}}};
}

ConvTransposed2dObj::ConvTransposed2dObj(GraphObj *graph, Tensor input,
                                         Tensor weight, Tensor output, int ph,
                                         int pw, int sh, int sw, int dh, int dw,
                                         int oph, int opw, int group,
                                         Tensor bias, ActType act)
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, ph, pw, sh,
                  sw, dh, dw, output, weight, act),
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
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, mode, sh, sw,
                  dh, dw, output, weight, act),
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
    : ConvBaseObj(OpType::ConvTranspose, {input, weight}, output, mode, sh, sw,
                  dh, dw, output, weight, act),
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
