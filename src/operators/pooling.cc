#include "operators/pooling.h"

namespace infini {

PoolingObj::PoolingObj(GraphObj *graph, OpType optype, Tensor input,
                       Tensor output, int kh, int kw, int dh, int dw, int ph,
                       int pw, int sh, int sw)
    : OperatorObj(optype, {input}, {output}),

      kh(kh), kw(kw), dh(dh), dw(dw), ph(ph), pw(pw), sh(sh), sw(sw),

      n(input->getDims()[0]), c(input->getDims()[1]), h(input->getDims()[2]),
      w(input->getDims()[3]) {
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> PoolingObj::inferShape(const TensorVec &inputs) const {
    const auto &input = inputs[0];
    auto h = input->getDims()[input->getDims().size() - 2],
         w = input->getDims()[input->getDims().size() - 1];
    int oh = (h - (kh - sh) + ph * 2) / sh;
    int ow = (w - (kw - sw) + pw * 2) / sw;
    auto ret = input->getDims();
    ret[input->getDims().size() - 2] = oh;
    ret[input->getDims().size() - 1] = ow;
    return {{ret}};
}

std::string PoolingObj::toString() const {
    std::ostringstream os;
    os << OpRegistry::getOpName(type) << "[" << getGuid() << "]";
    os << "(";
    os << "k=[" << kh << "," << kw << "],";
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

vector<int> PoolingObj::getWorkloadVector() const {
    return {
        enum_to_underlying(type), n, c, h, w, kh, kw, ph, pw, sh, sw, dh, dw};
}

vector<int> PoolingObj::getOpAttrVector() const {
    return {enum_to_underlying(type), kh, kw, ph, pw, sh, sw, dh, dw};
}

}; // namespace infini
