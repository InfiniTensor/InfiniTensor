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

optional<vector<Shape>> PoolingObj::inferShape(const TensorVec &inputs) {
    const auto &input = inputs[0];
    const auto &dims = input->getDims();
    IT_ASSERT(dims.size() == 3 || dims.size() == 4);
    n = dims[0];
    c = dims[1];
    h = dims.size() == 3 ? 1 : dims[2];
    w = dims.back();
    const auto outputSize = [&](int length, int kernel, int dilation,
                                int padding, int stride) {
        IT_ASSERT(kernel > 0 && dilation > 0 && stride > 0 && padding >= 0);
        const double size = (int64_t(length) + 2LL * padding -
                             int64_t(dilation) * (kernel - 1) - 1) /
                                double(stride) +
                            1;
        int result = ceilMode ? std::ceil(size) : std::floor(size);
        if (ceilMode &&
            int64_t(result - 1) * stride >= int64_t(length) + padding)
            --result;
        IT_ASSERT(result > 0, "Pooling output dimensions must be positive");
        return result;
    };
    const int ow = outputSize(w, kw, dw, pw, sw);

    auto ret = input->getDims();
    if (input->getRank() == 4) {
        ret[input->getRank() - 2] = outputSize(h, kh, dh, ph, sh);
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
