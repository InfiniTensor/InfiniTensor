#include "operators/convswish.h"

namespace infini {
ConvSwishObj::ConvSwishObj(GraphObj *graph, Tensor input, Tensor weight,
                           Tensor bias, Tensor output, vector<int> pad_,
                           vector<int> stride_, vector<int> dilation_)
    : OperatorObj(OpType::ConvSwish,
                  bias != nullptr ? std::vector<Tensor>{input, weight, bias}
                                  : std::vector<Tensor>{input, weight},
                  {output}),
      pad(pad_), stride(stride_), dilation(dilation_) {
    IT_ASSERT(pad.size() == 4);
    IT_ASSERT(stride.size() == 2);
    IT_ASSERT(dilation.size() == 2);
    IT_ASSERT(checkValid(graph));
}

std::string ConvSwishObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getDims()) << ",";
    os << vecToString(inputs[1]->getDims()) << ",";
    if (inputs.size() == 3) {
        os << vecToString(inputs[2]->getDims()) << ",";
    }
    os << "p= " << vecToString(pad) << " ,";
    os << "s= " << vecToString(stride) << " ,";
    os << "d= " << vecToString(dilation) << " ,";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    if (inputs.size() == 3) {
        os << "bias=" << inputs[2]->getGuid() << ",";
    }
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<Shape>> ConvSwishObj::inferShape(const TensorVec &inputs) {
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
    int sh = stride[0];
    int sw = stride[1];
    int dh = dilation[0];
    int dw = dilation[1];
    int ph = pad[0];
    int pw = pad[1];
    oh = (h - (r - sh) * dh + ph * 2) / sh;
    ow = (w - (s - sw) * dw + pw * 2) / sw;
    return {{{on, oc, oh, ow}}};
}

vector<int> ConvSwishObj::getWorkloadVector() const {
    return {type.underlying(),
            n,
            c,
            h,
            w,
            f,
            r,
            s,
            pad[0],
            pad[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1]};
}

vector<int> ConvSwishObj::getOpAttrVector() const {
    // IT_TODO_HALT(); // should padding mode / ph+pw be in attrs?
    return {type.underlying(),
            c,
            f,
            r,
            s,
            pad[0],
            pad[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1]};
}
} // namespace infini