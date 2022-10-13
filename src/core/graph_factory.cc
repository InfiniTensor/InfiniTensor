#include "core/graph_factory.h"

namespace infini {

Operator GraphFactoryObj::conv(Tensor input, Tensor weight, Tensor output,
                               int ph, int pw, int sh, int sw, int dh, int dw,
                               Tensor bias) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ConvObj>(i0, w0, o0, ph, ph, sh, sw, dh, dw, bias);
    return op;
}

Operator GraphFactoryObj::conv(Tensor input, Tensor weight, int ph, int pw,
                               int sh, int sw, int dh, int dw, Tensor bias) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    auto op = g->addOp<ConvObj>(i0, w0, nullptr, ph, ph, sh, sw, dh, dw, bias);
    return op;
}

Operator GraphFactoryObj::conv(Tensor input, Tensor weight, Tensor output,
                               ConvBaseObj::PaddingMode pm, int sh, int sw,
                               int dh, int dw, Tensor bias) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ConvObj>(i0, w0, o0, pm, sh, sw, dh, dw, bias);
    return op;
}

Operator GraphFactoryObj::conv(Tensor input, Tensor weight,
                               ConvBaseObj::PaddingMode pm, int sh, int sw,
                               int dh, int dw, Tensor bias) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    auto op = g->addOp<ConvObj>(i0, w0, nullptr, pm, sh, sw, dh, dw, bias);
    return op;
}

Operator GraphFactoryObj::matmul(Tensor A, Tensor B, Tensor C, bool transA,
                                 bool transB, Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(A->getDims(), A->getDType());
    Tensor i1 = g->addTensor(B->getDims(), B->getDType());
    Tensor o0 = g->addTensor(C->getDims(), C->getDType());
    auto op = g->addOpWithOutputs<MatmulObj>(i0, i1, o0, transA, transB, bias, act);
    return op;
}

Operator GraphFactoryObj::matmul(Tensor A, Tensor B, bool transA, bool transB,
                                 Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(A->getDims(), A->getDType());
    Tensor i1 = g->addTensor(B->getDims(), B->getDType());
    auto op = g->addOp<MatmulObj>(i0, i1, nullptr, transA, transB, bias, act);
    return op;
}

Operator GraphFactoryObj::convTrans(Tensor input, Tensor weight, Tensor output,
                                    int ph, int pw, int sh, int sw, int dh,
                                    int dw, int oph, int opw, int group,
                                    Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ConvTransposed2dObj>(i0, w0, o0, ph, pw, sh, sw, dh, dw,
                                            oph, opw, group, bias, act);
    return op;
}

Operator GraphFactoryObj::convTrans(Tensor input, Tensor weight, int ph, int pw,
                                    int sh, int sw, int dh, int dw, int oph,
                                    int opw, int group, Tensor bias,
                                    ActType act) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    auto op = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, ph, pw, sh, sw, dh,
                                            dw, oph, opw, group, bias, act);
    return op;
}

Operator GraphFactoryObj::convTrans(Tensor input, Tensor weight, Tensor output,
                                    ConvBaseObj::PaddingMode pm, int sh, int sw,
                                    int dh, int dw, int oph, int opw, int group,
                                    Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ConvTransposed2dObj>(i0, w0, o0, pm, sh, sw, dh, dw, oph,
                                            opw, group, bias, act);
    return op;
}

Operator GraphFactoryObj::convTrans(Tensor input, Tensor weight,
                                    ConvBaseObj::PaddingMode pm, int sh, int sw,
                                    int dh, int dw, int oph, int opw, int group,
                                    Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor w0 = g->addTensor(weight->getDims(), weight->getDType());
    auto op = g->addOp<ConvTransposed2dObj>(i0, w0, nullptr, pm, sh, sw, dh, dw,
                                            oph, opw, group, bias, act);
    return op;
}

Operator GraphFactoryObj::g2bmm(Tensor A, Tensor B, Tensor C, const int width,
                                const int dilation, Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(A->getDims(), A->getDType());
    Tensor i1 = g->addTensor(B->getDims(), B->getDType());
    Tensor o0 = g->addTensor(C->getDims(), C->getDType());
    auto op = g->addOpWithOutputs<G2BMMObj>(i0, i1, o0, width, dilation, bias, act);
    return op;
}

Operator GraphFactoryObj::g2bmm(Tensor A, Tensor B, const int width,
                                const int dilation, Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(A->getDims(), A->getDType());
    Tensor i1 = g->addTensor(B->getDims(), B->getDType());
    auto op = g->addOp<G2BMMObj>(i0, i1, nullptr, width, dilation, bias, act);
    return op;
}

Operator GraphFactoryObj::gbmml(Tensor A, Tensor B, Tensor C,
                                const int dilation, Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(A->getDims(), A->getDType());
    Tensor i1 = g->addTensor(B->getDims(), B->getDType());
    Tensor o0 = g->addTensor(C->getDims(), C->getDType());
    auto op = g->addOpWithOutputs<GBMMObj>(i0, i1, o0, dilation, bias, act);
    return op;
}

Operator GraphFactoryObj::gbmml(Tensor A, Tensor B, const int dilation,
                                Tensor bias, ActType act) {
    Tensor i0 = g->addTensor(A->getDims(), A->getDType());
    Tensor i1 = g->addTensor(B->getDims(), B->getDType());
    auto op = g->addOp<GBMMObj>(i0, i1, nullptr, dilation, bias, act);
    return op;
}

Operator GraphFactoryObj::pad(Tensor input, Tensor output,
                              const vector<int> &pads,
                              const optional<const vector<int>> &axis) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<PadObj>(i0, o0, pads, axis);
    return op;
}

Operator GraphFactoryObj::pad(Tensor input, const vector<int> &pads,
                              const optional<const vector<int>> &axis) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<PadObj>(i0, nullptr, pads, axis);
    return op;
}

Operator GraphFactoryObj::slice(Tensor input, Tensor output,
                                const vector<int> &starts,
                                const vector<int> &ends,
                                const optional<vector<int>> &axis,
                                const optional<vector<int>> &steps) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<SliceObj>(i0, o0, starts, ends, axis, steps);
    return op;
}

Operator GraphFactoryObj::slice(Tensor input, const vector<int> &starts,
                                const vector<int> &ends,
                                const optional<vector<int>> &axis,
                                const optional<vector<int>> &steps) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<SliceObj>(i0, nullptr, starts, ends, axis, steps);
    return op;
}

Operator GraphFactoryObj::concat(TensorVec inputs, Tensor output, int dim) {
    TensorVec is;
    for (auto input : inputs) {
        Tensor i = g->addTensor(input->getDims(), input->getDType());
        is.push_back(i);
    }
    Tensor o = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ConcatObj>(is, o, dim);
    return op;
}

Operator GraphFactoryObj::concat(TensorVec inputs, int dim) {
    TensorVec is;
    for (auto input : inputs) {
        Tensor i = g->addTensor(input->getDims(), input->getDType());
        is.push_back(i);
    }
    auto op = g->addOp<ConcatObj>(is, nullptr, dim);
    return op;
}

Operator GraphFactoryObj::split(Tensor input, std::optional<TensorVec> outputs,
                                int dim, int num) {
    Tensor i = g->addTensor(input->getDims(), input->getDType());
    if (outputs.has_value()) {
        TensorVec os;
        for (auto output : outputs.value()) {
            Tensor o = g->addTensor(output->getDims(), output->getDType());
            os.push_back(o);
        }
        auto op = g->addOpWithOutputs<SplitObj>(i, os, dim, num);
        return op;
    } else {
        auto op = g->addOp<SplitObj>(i, std::nullopt, dim, num);
        return op;
    }
}

Operator GraphFactoryObj::split(Tensor input, int dim, int num) {
    Tensor i = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<SplitObj>(i, std::nullopt, dim, num);
    return op;
}

Operator GraphFactoryObj::split(Tensor input, std::optional<TensorVec> outputs,
                                int dim, const vector<int> &ratio) {
    Tensor i = g->addTensor(input->getDims(), input->getDType());
    if (outputs.has_value()) {
        TensorVec os;
        for (auto output : outputs.value()) {
            Tensor o = g->addTensor(output->getDims(), output->getDType());
            os.push_back(o);
        }
        auto op = g->addOpWithOutputs<SplitObj>(i, os, dim, ratio);
        return op;
    } else {
        auto op = g->addOp<SplitObj>(i, std::nullopt, dim, ratio);
        return op;
    }
}

Operator GraphFactoryObj::split(Tensor input, int dim,
                                const vector<int> &ratio) {
    Tensor i = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<SplitObj>(i, std::nullopt, dim, ratio);
    return op;
}

Operator GraphFactoryObj::extend(Tensor input, Tensor output, int dim,
                                 int num) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ExtendObj>(i0, o0, dim, num);
    return op;
}

Operator GraphFactoryObj::extend(Tensor input, int dim, int num) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<ExtendObj>(i0, nullptr, dim, num);
    return op;
}

Operator GraphFactoryObj::maxpool(Tensor input, Tensor output, int kh, int kw,
                                  int dh, int dw, int ph, int pw, int sh,
                                  int sw) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<MaxPoolObj>(i0, o0, kh, kw, dh, dw, ph, pw, sh, sw);
    return op;
}

Operator GraphFactoryObj::maxpool(Tensor input, int kh, int kw, int dh, int dw,
                                  int ph, int pw, int sh, int sw) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<MaxPoolObj>(i0, nullptr, kh, kw, dh, dw, ph, pw, sh, sw);
    return op;
}

Operator GraphFactoryObj::avgpool(Tensor input, Tensor output, int kh, int kw,
                                  int dh, int dw, int ph, int pw, int sh,
                                  int sw) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOpWithOutputs<AvgPoolObj>(i0, o0, kh, kw, dh, dw, ph, pw, sh, sw);
    return op;
}

Operator GraphFactoryObj::avgpool(Tensor input, int kh, int kw, int dh, int dw,
                                  int ph, int pw, int sh, int sw) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<AvgPoolObj>(i0, nullptr, kh, kw, dh, dw, ph, pw, sh, sw);
    return op;
}

Operator GraphFactoryObj::add(Tensor input0, Tensor input1, Tensor output) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<AddObj>(i0, i1, o0);
    return op;
}

Operator GraphFactoryObj::add(Tensor input0, Tensor input1) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    auto op = g->addOp<AddObj>(i0, i1, nullptr);
    return op;
}

Operator GraphFactoryObj::sub(Tensor input0, Tensor input1, Tensor output) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<SubObj>(i0, i1, o0);
    return op;
}

Operator GraphFactoryObj::sub(Tensor input0, Tensor input1) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    auto op = g->addOp<SubObj>(i0, i1, nullptr);
    return op;
}

Operator GraphFactoryObj::mul(Tensor input0, Tensor input1, Tensor output) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<MulObj>(i0, i1, o0);
    return op;
}

Operator GraphFactoryObj::mul(Tensor input0, Tensor input1) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    auto op = g->addOp<SubObj>(i0, i1, nullptr);
    return op;
}

Operator GraphFactoryObj::div(Tensor input0, Tensor input1, Tensor output) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<DivObj>(i0, i1, o0);
    return op;
}

Operator GraphFactoryObj::div(Tensor input0, Tensor input1) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    auto op = g->addOp<DivObj>(i0, i1, nullptr);
    return op;
}

Operator GraphFactoryObj::pow(Tensor input0, Tensor input1, Tensor output) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<PowObj>(i0, i1, o0);
    return op;
}

Operator GraphFactoryObj::pow(Tensor input0, Tensor input1) {
    Tensor i0 = g->addTensor(input0->getDims(), input0->getDType());
    Tensor i1 = g->addTensor(input1->getDims(), input1->getDType());
    auto op = g->addOp<PowObj>(i0, i1, nullptr);
    return op;
}

Operator GraphFactoryObj::gather(Tensor input, Tensor index, Tensor output,
                                 int axis) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<GatherObj>(i0, index, o0, axis);
    return op;
}

Operator GraphFactoryObj::gather(Tensor input, Tensor index, int axis) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<GatherObj>(i0, index, nullptr, axis);
    return op;
}

Operator GraphFactoryObj::reshape(Tensor input, Tensor output,
                                  const Shape &dims) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ReshapeObj>(i0, o0, dims);
    return op;
}

Operator GraphFactoryObj::reshape(Tensor input, const Shape &dims) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<ReshapeObj>(i0, nullptr, dims);
    return op;
}

Operator GraphFactoryObj::flatten(Tensor input, Tensor output) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<FlattenObj>(i0, o0);
    return op;
}

Operator GraphFactoryObj::flatten(Tensor input) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<FlattenObj>(i0, nullptr);
    return op;
}

Operator GraphFactoryObj::identity(Tensor input, Tensor output) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<IdentityObj>(i0, o0);
    return op;
}

Operator GraphFactoryObj::identity(Tensor input) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<IdentityObj>(i0, nullptr);
    return op;
}

Operator GraphFactoryObj::softmax(Tensor input, Tensor output) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<SoftmaxObj>(i0, o0);
    return op;
}

Operator GraphFactoryObj::softmax(Tensor input) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<SoftmaxObj>(i0, nullptr);
    return op;
}

Operator GraphFactoryObj::relu(Tensor input, Tensor output) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<ReluObj>(i0, o0);
    return op;
}

Operator GraphFactoryObj::relu(Tensor input) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<ReluObj>(i0, nullptr);
    return op;
}

Operator GraphFactoryObj::sigmoid(Tensor input, Tensor output) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<SigmoidObj>(i0, o0);
    return op;
}

Operator GraphFactoryObj::sigmoid(Tensor input) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<SigmoidObj>(i0, nullptr);
    return op;
}

Operator GraphFactoryObj::tanh(Tensor input, Tensor output) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<TanhObj>(i0, o0);
    return op;
}

Operator GraphFactoryObj::tanh(Tensor input) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<TanhObj>(i0, nullptr);
    return op;
}

Operator GraphFactoryObj::abs(Tensor input, Tensor output) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    Tensor o0 = g->addTensor(output->getDims(), output->getDType());
    auto op = g->addOpWithOutputs<AbsObj>(i0, o0);
    return op;
}

Operator GraphFactoryObj::abs(Tensor input) {
    Tensor i0 = g->addTensor(input->getDims(), input->getDType());
    auto op = g->addOp<AbsObj>(i0, nullptr);
    return op;
}

Operator GraphFactoryObj::memBound(const TensorVec &inputs,
                                   const TensorVec &outputs,
                                   const std::vector<nnet::Tensor> &nnetInputs,
                                   nnet::Expr expr, double exec_time,
                                   std::string hint) {
    TensorVec is;
    for (auto input : inputs) {
        auto i = g->addTensor(input->getDims(), input->getDType());
        is.push_back(i);
    }
    TensorVec os;
    for (auto output : outputs) {
        auto o = g->addTensor(output->getDims(), output->getDType());
        os.push_back(o);
    }
    auto op = g->addOpWithOutputs<MemBoundObj>(is, os, nnetInputs, expr, exec_time, hint);
    return op;
}

} // namespace infini