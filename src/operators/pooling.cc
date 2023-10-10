#include "operators/pooling.h"

namespace infini {

PoolingObj::PoolingObj(GraphObj *graph, OpType optype, Tensor input,
                       Tensor output, int kh, int kw, int dh, int dw, int ph,
                       int pw, int sh, int sw, int ceilMode)
    : OperatorObj(optype, {input}, {output}), kh(kh), kw(kw), dh(dh), dw(dw),
      ph(ph), pw(pw), sh(sh), sw(sw), ceilMode(ceilMode),
      n(input->getDims()[0]), c(input->getDims()[1]), h(input->getDims()[2]),
      w(input->getDims()[3]) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PoolingObj::inferShape(const TensorVec &inputs) const {
    const auto &input = inputs[0];
    auto h = input->getDims()[input->getRank() - 2],
         w = input->getDims()[input->getRank() - 1];
    int oh, ow;
    if (ceilMode) {
        oh = ceil(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = ceil(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    } else {
        oh = floor(((float)(h + 2 * ph - dh * (kh - 1) - 1)) / sh + 1);
        ow = floor(((float)(w + 2 * pw - dw * (kw - 1) - 1)) / sw + 1);
    }
    auto ret = input->getDims();
    ret[input->getRank() - 2] = oh;
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
