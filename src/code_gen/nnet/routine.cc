#include "code_gen/nnet/routine.h"
#include "code_gen/nnet/Visitor/CloneMutator.h"
#include <algorithm>
namespace nnet {

RoutineNode::RoutineNode(Expr _expr, const vector<Tensor> &_inputs)
    : inputs(_inputs) {
    if (_expr)
        expr = CloneMutator().clone(_expr);
    else
        expr = nullptr;
}

string MatmulNode::toReadable() const {
    std::ostringstream oss;
    assert(inputs.size() == 2);
    oss << "Matmul{bmnk = " << b << ", " << m << ", " << n << ", " << k
        << "; AB = " << inputs[0] << ", " << inputs[1]
        << "; transAB = " << transa << ", " << transb << "}";
    return oss.str();
}

bool operator==(const MatmulNode &lhs, const MatmulNode &rhs) {
    if (!((lhs.b == rhs.b) && lhs.m == rhs.m && lhs.n == rhs.n &&
          lhs.k == rhs.k && lhs.transa == rhs.transa &&
          lhs.transb == rhs.transb))
        return false;
    if (lhs.inputs.size() != rhs.inputs.size())
        return false;
    for (size_t i = 0; i < lhs.inputs.size(); ++i) {
        if (lhs.inputs[i]->getName() != rhs.inputs[i]->getName())
            return false;
    }
    return true;
}

string ConvNode::toReadable() const {
    std::ostringstream oss;
    assert(inputs.size() == 2);
    oss << "Conv{A =" << inputs[0]
        << " shape=" << serializeVec(inputs[0]->getShape())
        << ", K=" << inputs[1]
        << " shape=" << serializeVec(inputs[1]->getShape()) << ", p = " << ph
        << ", " << pw << ", s= " << sh << ", " << sw << ", d= " << dh << ", "
        << dw << "}";
    return oss.str();
}

bool operator==(const ConvNode &lhs, const ConvNode &rhs) {
    if (!(lhs.ph == rhs.ph && lhs.pw == rhs.pw && lhs.sh == rhs.sh &&
          lhs.sw == rhs.sw && lhs.dh == rhs.dh && lhs.dw == rhs.dw))
        return false;
    if (lhs.inputs.size() != rhs.inputs.size())
        return false;
    for (size_t i = 0; i < lhs.inputs.size(); ++i) {
        if (lhs.inputs[i]->getName() != rhs.inputs[i]->getName())
            return false;
    }
    return true;
}

vector<int> ConvNode::getShape() const {
    auto input = inputs[0], weight = inputs[1];
    auto n = input->getShape(0);
    auto h = input->getShape(2);
    auto w = input->getShape(3);
    auto f = weight->getShape(0);
    auto r = weight->getShape(2);
    auto s = weight->getShape(3);
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // Set padding size
    oh = (h - (r - sh) * dh + ph * 2) / sh;
    ow = (w - (s - sw) * dw + pw * 2) / sw;
    auto ret = {on, oc, oh, ow};
    return ret;
}

ConvArgs ConvNode::getArgs() const { return tuple(ph, pw, sh, sw, dh, dw); }

vector<int> G2bmmNode::getShape() const { return {b, m, 2 * w + 1}; }

vector<int> GbmmNode::getShape() const { return {b, m, n}; }

string ElementWiseNode::toReadable() const {
    std::ostringstream oss;
    oss << "EleWise{";
    for (const auto &input : inputs)
        oss << input << ", ";
    oss << "}";
    return oss.str();
}

double ElementWiseNode::getEstimatedTime() const {
    int64_t cntElements = 0;
    // For unimplemented transpose
    assert(inputs.size() > 0);
    if (!expr) {
        assert(inputs.size() == 1);
    }
    for (const auto &input : inputs)
        cntElements += input->getSize();
    int64_t outputSize = 1;
    for (const auto &len : outputShape)
        outputSize *= len;
    cntElements += outputSize;

    const double bandwidth = 200 * 1000000;
    // dbg(inputs, inputs[0]->getShape(), cntElements,
    // (cntElements * 4) / bandwidth);
    return double(cntElements * 4) / bandwidth; // ms
}

string G2bmmNode::toReadable() const {
    std::ostringstream oss;
    oss << "G2bmm{";
    for (const auto &input : inputs)
        oss << input << ", ";
    oss << ", bmwk = " << b << " " << m << " " << w << " " << k << "}";
    return oss.str();
}

string GbmmNode::toReadable() const {
    std::ostringstream oss;
    oss << "Gbmm{";
    for (const auto &input : inputs)
        oss << input << ", ";
    oss << ", bmwn = " << b << " " << m << " " << w << " " << n << "}";
    return oss.str();
}

G2bmmArgs G2bmmNode::getArgs() const { return {b, m, w, k, 1}; }

GbmmArgs GbmmNode::getArgs() const { return {b, m, w, n, 1}; }

} // namespace nnet