#include "code_gen/graph.h"
#include "code_gen/common.h"
#include "code_gen/operator.h"
#include "code_gen/tensor.h"
#include "ffi/ffi_embed.h"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tpm {

GraphBase::~GraphBase() {
    for (auto op : ops)
        if (op != nullptr)
            delete op;
    for (auto tensor : tensors)
        if (tensor != nullptr)
            delete tensor;
}

void GraphBase::addTensor(Tensor *tensor) { tensors.emplace_back(tensor); }

TensorVec &GraphBase::getTensors() { return tensors; }

OpVec &GraphBase::getOperators() { return ops; }
const OpVec &GraphBase::getOperators() const { return ops; }

TensorVec &GraphBase::getInputs() { return inputs; }
const TensorVec &GraphBase::getInputs() const { return inputs; }

TensorVec &GraphBase::getOutputs() { return outputs; }
const TensorVec &GraphBase::getOutputs() const { return outputs; }

void GraphBase::updateConnection() {
    for (auto op : ops) {
        for (auto tensor : op->getInputs())
            tensor->addInputOf(op);
        for (auto tensor : op->getOutputs())
            tensor->setOutputOf(op);
    }
    // update successors and predecessors for all ops
    for (auto tensor : tensors) {
        for (auto opNext : tensor->getInputOf()) {
            auto opPrev = tensor->getOutputOf();
            if (opPrev != nullptr) {
                opNext->addPredecessors(opPrev);
                opPrev->addSuccessors(opNext);
            }
        }
        if (!tensor->getInputOf().empty() && tensor->getOutputOf() == nullptr)
            inputs.emplace_back(tensor);
        if (tensor->getInputOf().empty() && tensor->getOutputOf() != nullptr)
            outputs.emplace_back(tensor);
    }
}

Operator *Graph::conv(Tensor *input, Tensor *weight, Tensor *output, int ph,
                      int pw, int sh, int sw, int dh, int dw, Tensor *bias) {
    auto op = new ConvOp(input, weight, output, ph, pw, sh, sw, dh, dw, bias);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::conv(Tensor *input, Tensor *weight, int ph, int pw, int sh,
                      int sw, int dh, int dw, Tensor *bias) {
    auto op = new ConvOp(input, weight, ph, pw, sh, sw, dh, dw, bias);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::conv(Tensor *input, Tensor *weight, Tensor *output,
                      ConvOp::PaddingMode pm, int sh, int sw, int dh, int dw,
                      Tensor *bias) {
    auto op = new ConvOp(input, weight, output, pm, sh, sw, dh, dw, bias);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::conv(Tensor *input, Tensor *weight, ConvOp::PaddingMode pm,
                      int sh, int sw, int dh, int dw, Tensor *bias) {
    auto op = new ConvOp(input, weight, pm, sh, sw, dh, dw, bias);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Tensor *Graph::setConvTransInput(Tensor *input) {
    auto op = new TransposeOp(input, -1, {0, 2, 3, 1});
    auto transInput = op->getOutputs()[0];
    ops.emplace_back(op);
    addTensor(transInput);
    return transInput;
}

Tensor *Graph::setConvTransWeight(Tensor *weight) {
    weight->setType(Tensor::Weight);
    auto op = new TransposeOp(weight, -1, {2, 3, 0, 1});
    auto transWeight = op->getOutputs()[0];
    ops.emplace_back(op);
    addTensor(transWeight);
    return transWeight;
}

Tensor *Graph::setConvTransOutput(Tensor *outputTrans, Tensor *output) {
    if (output == nullptr) {
        auto op = new TransposeOp(outputTrans, -1, {0, 3, 1, 2});
        auto output = op->getOutputs()[0];
        ops.emplace_back(op);
        addTensor(output);
    } else {
        auto op = new TransposeOp(outputTrans, output, -1, {0, 3, 1, 2});
        ops.emplace_back(op);
    }
    return output;
}

Operator *Graph::convTrans(Tensor *input, Tensor *weight, Tensor *output,
                           int ph, int pw, int sh, int sw, int dh, int dw,
                           int oph, int opw, Tensor *bias) {
    input = setConvTransInput(input);
    weight = setConvTransWeight(weight);
    auto op =
        new ConvTransOp(input, weight, ph, pw, sh, sw, dh, dw, oph, opw, bias);
    ops.emplace_back(op);
    auto outputTrans = op->getOutputs()[0];
    setConvTransOutput(outputTrans, output);
    return op;
}

Operator *Graph::convTrans(Tensor *input, Tensor *weight, int ph, int pw,
                           int sh, int sw, int dh, int dw, int oph, int opw,
                           Tensor *bias) {
    // Not a correct constructor
    assert(false);
    input = setConvTransInput(input);
    weight = setConvTransWeight(weight);
    auto op =
        new ConvTransOp(input, weight, ph, pw, sh, sw, dh, dw, oph, opw, bias);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    setConvTransOutput(output);
    return op;
}

Operator *Graph::convTrans(Tensor *input, Tensor *weight, Tensor *output,
                           ConvTransOp::PaddingMode pm, int sh, int sw, int dh,
                           int dw, int oph, int opw, Tensor *bias) {
    input = setConvTransInput(input);
    weight = setConvTransWeight(weight);
    auto op =
        new ConvTransOp(input, weight, pm, sh, sw, dh, dw, oph, opw, bias);
    ops.emplace_back(op);
    auto outputTrans = op->getOutputs()[0];
    setConvTransOutput(outputTrans, output);
    return op;
}

Operator *Graph::convTrans(Tensor *input, Tensor *weight,
                           ConvTransOp::PaddingMode pm, int sh, int sw, int dh,
                           int dw, int oph, int opw, Tensor *bias) {
    // Not a correct constructor
    assert(false);
    input = setConvTransInput(input);
    weight = setConvTransWeight(weight);
    auto op =
        new ConvTransOp(input, weight, pm, sh, sw, dh, dw, oph, opw, bias);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    setConvTransOutput(output);
    return op;
}

Operator *Graph::matmul(Tensor *A, Tensor *B, Tensor *C, bool transA,
                        bool transB, Tensor *bias) {
    auto op = new MatmulOp(A, B, C, transA, transB, bias);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::matmul(Tensor *A, Tensor *B, bool transA, bool transB,
                        Tensor *bias) {
    auto op = new MatmulOp(A, B, transA, transB, bias);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::g2bmm(Tensor *A, Tensor *B, Tensor *C, int width, int dilation,
                       Tensor *bias) {
    auto op = new G2BMMOp(A, B, C, width, dilation, bias);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::g2bmm(Tensor *A, Tensor *B, int width, int dilation,
                       Tensor *bias) {
    auto op = new G2BMMOp(A, B, width, dilation, bias);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::gbmml(Tensor *A, Tensor *B, Tensor *C, int dilation,
                       Tensor *bias) {
    auto op = new GBMMLOp(A, B, C, dilation, bias);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::gbmml(Tensor *A, Tensor *B, int dilation, Tensor *bias) {
    auto op = new GBMMLOp(A, B, dilation, bias, Operator::ActType::None);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::pad(Tensor *input, Tensor *output, const Dim &begin,
                     const Dim &end) {
    auto op = new PadOp(input, output, begin, end);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::pad(Tensor *input, const Dim &begin, const Dim &end) {
    auto op = new PadOp(input, begin, end);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::slice(Tensor *input, Tensor *output, const Dim &begin,
                       const Dim &end) {
    auto op = new SliceOp(input, output, begin, end);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::slice(Tensor *input, const Dim &begin, const Dim &end) {
    auto op = new SliceOp(input, begin, end);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::slice(Tensor *input, Tensor *output, Tensor *begin,
                       Tensor *end) {
    Dim _begin(begin->size());
    auto begin_d = begin->getDataPtr();
    for (size_t i = 0, iEnd = begin->size(); i < iEnd; i++) {
        if (begin_d != nullptr)
            _begin[i] = begin_d[i];
        else
            _begin[i] = 0;
    }
    Dim _end(end->size());
    auto end_d = end->getDataPtr();
    for (size_t i = 0, iEnd = end->size(); i < iEnd; i++) {
        if (end_d != nullptr)
            _end[i] = end_d[i];
        else
            _end[i] = 0;
    }
    auto op = new SliceOp(input, output, _begin, _end);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::concat(const TensorVec &inputs, Tensor *output, int dim) {
    auto op = new ConcatOp(inputs, output, dim);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::concat(const TensorVec &inputs, int dim) {
    auto op = new ConcatOp(inputs, dim);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::split(Tensor *input, const TensorVec &outputs, int dim,
                       int num) {
    auto op = new SplitOp(input, outputs, dim, num);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::split(Tensor *input, int dim, int num) {
    auto op = new SplitOp(input, dim, num);
    ops.emplace_back(op);
    auto outputs = op->getOutputs();
    for (auto output : outputs)
        addTensor(output);
    return op;
}

Operator *Graph::split(Tensor *input, const TensorVec &outputs, int dim,
                       std::vector<int> sizes) {
    auto op = new SplitOp(input, outputs, dim, sizes);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::split(Tensor *input, int dim, std::vector<int> sizes) {
    auto op = new SplitOp(input, dim, sizes);
    ops.emplace_back(op);
    auto outputs = op->getOutputs();
    for (auto output : outputs)
        addTensor(output);
    return op;
}

Operator *Graph::transpose(Tensor *input, Tensor *output, int split,
                           const Perm &after, int factor) {
    if (input->getType() == Tensor::Weight)
        output->setType(Tensor::Weight);
    auto op = new TransposeOp(input, output, split, after, factor);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::transpose(Tensor *input, int split, const Perm &after,
                           int factor) {
    auto op = new TransposeOp(input, split, after, factor);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::flatten(Tensor *input, Tensor *output, int axis) {
    assert(axis > 0 && axis < (int)input->getDims().size());
    Dim d1, d2;
    for (size_t i = 0, iEnd = input->getDims().size(); i < iEnd; i++) {
        ((int)i < axis ? d1 : d2).emplace_back(i);
    }
    return transpose(input, output, -1, {d1, d2});
}

Operator *Graph::flatten(Tensor *input, int axis) {
    assert(axis > 0 && axis < (int)input->getDims().size());
    Dim d1, d2;
    for (size_t i = 0, iEnd = input->getDims().size(); i < iEnd; i++) {
        ((int)i < axis ? d1 : d2).emplace_back(i);
    }
    return transpose(input, -1, {d1, d2});
}

Operator *Graph::extend(Tensor *input, Tensor *output, int dim, int num) {
    auto op = new ExtendOp(input, output, dim, num);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::extend(Tensor *input, int dim, int num) {
    auto op = new ExtendOp(input, dim, num);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::batchnorm(Tensor *input, Tensor *scale, Tensor *bias,
                           Tensor *mean, Tensor *var, Tensor *output,
                           float epsilon, float momentum) {
    auto op = new BatchNormOp(input, scale, bias, mean, var, output, epsilon,
                              momentum);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::batchnorm(Tensor *input, Tensor *scale, Tensor *bias,
                           Tensor *mean, Tensor *var, float epsilon,
                           float momentum) {
    auto op = new BatchNormOp(input, scale, bias, mean, var, epsilon, momentum);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::maxpool(Tensor *input, Tensor *output, int kh, int kw, int dh,
                         int dw, int ph, int pw, int sh, int sw) {
    auto op = new MaxPoolOp(input, output, kh, kw, dh, dw, ph, pw, sh, sw);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::maxpool(Tensor *input, int kh, int kw, int dh, int dw, int ph,
                         int pw, int sh, int sw) {
    auto op = new MaxPoolOp(input, kh, kw, dh, dw, ph, pw, sh, sw);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::avgpool(Tensor *input, Tensor *output) {
    auto op = new AvgPoolOp(input, output);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::avgpool(Tensor *input, Tensor *output, int kh, int kw, int ph,
                         int pw, int sh, int sw) {
    auto op = new AvgPoolOp(input, output, kh, kw, ph, pw, sh, sw);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::avgpool(Tensor *input, int kh, int kw, int ph, int pw, int sh,
                         int sw) {
    auto op = new AvgPoolOp(input, kh, kw, ph, pw, sh, sw);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::add(const TensorVec &inputs, Tensor *output) {
    auto op = new AddOp(inputs, output);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::add(const TensorVec &inputs) {
    auto op = new AddOp(inputs);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::sub(Tensor *input0, Tensor *input1, Tensor *output) {
    auto op = new SubOp({input0, input1}, output);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::sub(Tensor *input0, Tensor *input1) {
    auto op = new SubOp({input0, input1});
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::mul(const TensorVec &inputs, Tensor *output) {
    auto op = new MulOp(inputs, output);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::mul(const TensorVec &inputs) {
    auto op = new MulOp(inputs);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::div(Tensor *input0, Tensor *input1, Tensor *output) {
    auto op = new DivOp({input0, input1}, output);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::div(Tensor *input0, Tensor *input1) {
    auto op = new DivOp({input0, input1});
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::pow(Tensor *input, Tensor *output, int pow) {
    auto op = new PowOp(input, output, pow);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::pow(Tensor *input, int pow) {
    auto op = new PowOp(input, pow);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::gather(Tensor *data, Tensor *indices, Tensor *output,
                        int axis) {
    auto op = new GatherOp(data, indices, output, axis);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::gather(Tensor *data, Tensor *indices, int axis) {
    auto op = new GatherOp(data, indices, axis);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::reduceMean(Tensor *input, Tensor *output, int axis) {
    auto op = new ReduceMeanOp(input, output, axis);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::reduceMean(Tensor *input, int axis) {
    auto op = new ReduceMeanOp(input, axis);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::reshape(Tensor *input, Tensor *output) {
    auto op = new ReshapeOp(input, output);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::identity(Tensor *input, Tensor *output) {
    auto op = new IdentityOp(input, output);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::identity(Tensor *input) {
    auto op = new IdentityOp(input);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::relu(Tensor *input, Tensor *output) {
    auto op = new ActivationOp(input, output, Operator::Relu);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::relu(Tensor *input) {
    auto op = new ActivationOp(input, Operator::Relu);
    ops.emplace_back(op);
    addTensor(op->getOutputs()[0]);
    return op;
}

Operator *Graph::sigmoid(Tensor *input, Tensor *output) {
    auto op = new ActivationOp(input, output, Operator::Sigmoid);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::sigmoid(Tensor *input) {
    auto op = new ActivationOp(input, Operator::Sigmoid);
    ops.emplace_back(op);
    addTensor(op->getOutputs()[0]);
    return op;
}

Operator *Graph::softmax(Tensor *input, Tensor *output, int axis) {
    auto op = new SoftmaxOp(input, output, axis);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::softmax(Tensor *input, int axis) {
    auto op = new SoftmaxOp(input, axis);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Operator *Graph::tanh(Tensor *input, Tensor *output) {
    auto op = new ActivationOp(input, output, Operator::Tanh);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::tanh(Tensor *input) {
    auto op = new ActivationOp(input, Operator::Tanh);
    ops.emplace_back(op);
    addTensor(op->getOutputs()[0]);
    return op;
}

Operator *Graph::membound(const TensorVec &inputs, const TensorVec &outputs,
                          const std::vector<nnet::Tensor> &nnetInputs,
                          nnet::Expr expr, double _exec_time,
                          std::string hint) {
    // 0.001 for kenrel launch
    double exec_time = (_exec_time == 0) ? 0 : std::max(_exec_time, 0.001);
    auto op =
        new MemBoundOp(inputs, outputs, nnetInputs, expr, exec_time, hint);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::resize(Tensor *input, Tensor *sizes, Tensor *output) {
    auto op = new ResizeOp(input, output, sizes);
    ops.emplace_back(op);
    return op;
}

Operator *Graph::resize(Tensor *input, Tensor *sizes) {
    auto op = new ResizeOp(input, sizes);
    ops.emplace_back(op);
    auto output = op->getOutputs()[0];
    addTensor(output);
    return op;
}

Tensor *GraphBase::tensor(const Dim &dims, Tensor::DataType dtype) {
    auto tensor = new Tensor(dims, Tensor::Input, dtype);
    tensors.emplace_back(tensor);
    return tensor;
}

Tensor *GraphBase::tensor(const Dim &dims, const std::string &dtype) {
    if (dtype == "FLOAT")
        return tensor(dims, Tensor::Float32);
    if (dtype == "INT32")
        return tensor(dims, Tensor::Int32);
    if (dtype == "INT64") // FIXME: Treat int64 as int32
        return tensor(dims, Tensor::Int32);
    std::cout << "Unsupported data type: " + dtype << std::endl;
    assert(false);
    return nullptr;
}

void Graph::setInputs(TensorVec inputs_) { inputs = inputs_; }

void Graph::setOutputs(TensorVec outputs_) { outputs = outputs_; }

bool Graph::importOnnx(const char *net) {
    infini::start_interpreter();
    try {
        py::module::import("cpp_plugin").attr("import_onnx")(this, net);
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_ImportError)) {
            std::cerr << "Import Error. Don't forget to set environment "
                         "variable PYTHONPATH to contain "
                         "<repo-root>/python"
                      << std::endl;
        }
        throw;
    }
    if (mutateInceptionHead())
        puts("Detecting inception head");

    updateConnection();
    return true;
}

void SubGraph::cleanConnection() {
    for (auto tensor : tensors) {
        tensor->setOutputOf(nullptr);
        tensor->setInputOf({});
    }
    for (auto op : ops) {
        op->setPredecessors({});
        op->setSuccessors({});
    }
    inputs.clear();
    outputs.clear();
}

int SubGraph::findTensor(Tensor *tensor, int ntensor) {
    ntensor = ntensor == 0 ? tensors.size() : ntensor;
    for (int i = 0; i < ntensor; ++i)
        if (tensors[i] == tensor)
            return i;
    return -1;
}

SubGraph::SubGraph(OpVec oplist) {
    hash = 2147483647;
    std::map<size_t, Tensor *> tensorMap;
    for (auto originOp : oplist) {
        auto op = originOp->clone();
        ops.emplace_back(op);
        for (auto tensor : originOp->getInputs()) {
            Tensor *t;
            if (tensorMap.find(tensor->getHash()) == tensorMap.end()) {
                t = tensor->clone();
                tensors.emplace_back(t);
                tensorMap.emplace(tensor->getHash(), t);
            } else
                t = tensorMap[tensor->getHash()];
            t->addInputOf(op);
            op->addInput(t);
        }
        for (auto tensor : originOp->getOutputs()) {
            Tensor *t;
            if (tensorMap.find(tensor->getHash()) == tensorMap.end()) {
                t = tensor->clone();
                tensors.emplace_back(t);
                tensorMap.emplace(tensor->getHash(), t);
            } else
                t = tensorMap[tensor->getHash()];
            t->setOutputOf(op);
            op->addOutput(t);
        }
    }
    for (auto tensor : tensors) {
        for (auto opNext : tensor->getInputOf()) {
            auto opPrev = tensor->getOutputOf();
            if (opPrev != nullptr) {
                opNext->addPredecessors(opPrev);
                opPrev->addSuccessors(opNext);
            }
        }
        if (!tensor->getInputOf().empty() && tensor->getOutputOf() == nullptr)
            inputs.emplace_back(tensor);
        if (tensor->getInputOf().empty() && tensor->getOutputOf() != nullptr)
            outputs.emplace_back(tensor);
    }
}

bool SubGraph::resetOps(OpVec oplist, size_t ntensor) {
    ntensor = ntensor == 0 ? tensors.size() : ntensor;
    hash = 2147483647;
    ops.clear();
    cleanConnection();

    for (auto op : oplist) {
        ops.emplace_back(op);
        op->setPredecessors({});
        op->setSuccessors({});
        for (auto tensor : op->getInputs()) {
            auto idx = findTensor(tensor);
            if (idx < 0)
                return false;
            auto t = tensors[idx];
            t->addInputOf(op);
        }
        for (auto tensor : op->getOutputs()) {
            auto idx = findTensor(tensor);
            if (idx < 0)
                return false;
            auto t = tensors[idx];
            t->setOutputOf(op);
        }
    }

    for (size_t i = 0; i < ntensor; ++i) {
        auto tensor = tensors[i];
        for (auto opNext : tensor->getInputOf()) {
            auto opPrev = tensor->getOutputOf();
            if (opPrev != nullptr) {
                opNext->addPredecessors(opPrev);
                opPrev->addSuccessors(opNext);
            }
        }
        if (!tensor->getInputOf().empty() && tensor->getOutputOf() == nullptr)
            inputs.emplace_back(tensor);
        if (tensor->getInputOf().empty() && tensor->getOutputOf() != nullptr)
            outputs.emplace_back(tensor);
    }
    return true;
}

const std::pair<bool, VType>
SubGraph::compute(const Dim &point, size_t outputId, bool getAllPos) const {
    if (outputId >= outputs.size()) {
        return {false, 0};
    }

    std::unordered_map<const Tensor *, DimRange> drs;
    for (auto output : outputs)
        drs[output] = DimRange::getEmpty();
    if (getAllPos)
        drs[outputs[outputId]] = DimRange::getAllPos();
    else
        drs[outputs[outputId]] = DimRange(point);
    // reversed DFS post-order is topo-order
    std::unordered_set<const Operator *> flag;
    // computing functions for operators
    std::vector<std::function<bool()>> runners;
    std::function<bool(Operator *)> dfs = [&](Operator *op) -> bool {
        if (flag.count(op)) {
            return true;
        }
        flag.insert(op);
        for (auto &&next : op->getSuccessors()) {
            if (!dfs(next)) {
                return false;
            }
        }
        std::vector<DimRange> inDrs;
        std::function<bool()> runner;
        Tensor *out = op->getOutput();
        if (out != nullptr) {
            op->print();
            std::tie(inDrs, runner) = op->compute(drs.at(out));
            if (runner == nullptr) {
                return false;
            }
        } else {
            out = op->getOutputs()[outputId];
            std::tie(inDrs, runner) =
                dynamic_cast<SplitOp *>(op)->compute(outputId, drs.at(out));
        }
        assert(op->isConcatOp() || (int)inDrs.size() == op->numInputs());
        for (size_t i = 0, iEnd = inDrs.size(); i < iEnd; i++) {
            const Tensor *t = op->getInputs()[i];
            drs[t] = drs.count(t) ? unionRange(drs[t], inDrs[i]) : inDrs[i];
        }
        runners.push_back(runner);
        return true;
    };
    // Operator *op = outputs[outputId]->getOutputOf();
    for (auto &&op : ops) {
        if (!dfs(op)) {
            return {false, 0};
        }
    }

    for (auto it = runners.rbegin(); it != runners.rend(); it++) {
        if (!(*it)()) {
            return {false, 0};
        }
    }
    return {true, outputs[outputId]->getData(point)};
}

uint64_t SubGraph::getHash() {
    if (hash != 2147483647) {
        return hash;
    }
    // TODO: Replace string operations with int operations
    auto &opList = getOperators();
    std::vector<int> cnt(opList.size());
    std::vector<uint64_t> nodeHash(opList.size());
    std::unordered_map<int, int> nodeMap;
    std::unordered_set<int> inputSet;
    std::vector<int> q;
    for (size_t i = 0; i < opList.size(); i++) {
        auto &op = opList[i];
        nodeMap.emplace(op->getGuid(), i);
        cnt[i] = op->getPredecessors().size();
        nodeHash[i] = hashPack(op->getHash());
        if (op->getPredecessors().size() == 0) {
            q.emplace_back(i);
        }
    }
    for (auto t : getInputs()) {
        inputSet.emplace(t->getGuid());
    }

    int st = 0, ed = q.size();
    while (st < ed) {
        int id = q[st];
        st++;
        auto &op = opList[id];
        for (auto t : op->getInputs()) {
            if (inputSet.find(t->getGuid()) != inputSet.end()) {
                nodeHash[id] = hashAppend(nodeHash[id], t->getHash());
            } else {
                nodeHash[id] =
                    hashAppend(nodeHash[id],
                               nodeHash[nodeMap[t->getOutputOf()->getGuid()]]);
            }
        }
        nodeHash[id] = hashPack(nodeHash[id]);
        for (auto suc : op->getSuccessors()) {
            int suc_id = nodeMap[suc->getGuid()];
            cnt[suc_id]--;
            if (cnt[suc_id] == 0) {
                q.emplace_back(suc_id);
                ed++;
            }
        }
    }

    hash = 0;
    for (auto t : getOutputs()) {
        int id = nodeMap[t->getOutputOf()->getGuid()];
        hash =
            hashAppend(hash, hashPack(hashAppend(t->getHash(), nodeHash[id])));
    }
    return hash;
}

uint64_t SubGraph::getCacheIdx(int level) {
    auto ops = getOperators();
    uint64_t hash = 0;
    // for all ops, the number is no more than 2 inputs * 4 dims
    uint64_t factors[8] = {10000019, 10000079, 10000103, 10000121,
                           10000139, 10000141, 10000169, 10000189};
    for (size_t i = 0; i < ops.size(); i++) {
        hash = hashAppend(hash, std::hash<uint64_t>()(ops[i]->getType()));
        if (level == 1) {
            for (size_t j = 0, l = 0; j < ops[i]->getInputs().size(); j++) {
                uint64_t tensorHash = 0;
                for (size_t k = 0; k < ops[i]->getInputs()[j]->getDims().size();
                     k++, l++) {
                    tensorHash = hashAppend(
                        tensorHash,
                        ops[i]->getInputs()[j]->getDims()[k] * factors[l]);
                }
                hash = hashAppend(hash, tensorHash);
            }
        }
    }
    return hash;
}

int SubGraph::print() {
    std::cout << "Subgraph[" << getHash() << "]" << std::endl;
    std::cout << "    op num: " << getOperators().size() << std::endl;
    std::cout << "    input: ";
    for (auto input : getInputs()) {
        std::cout << input->getHash() << " ";
    }
    std::cout << std::endl;
    std::cout << "    operator: " << std::endl;
    for (auto op : getOperators()) {
        std::cout << "        ";
        op->print();
        std::cout << "        ";
        std::cout << "[" << op->getGuid() << "]";
        std::cout << " pre=[";
        for (auto pre : op->getPredecessors()) {
            std::cout << pre->getGuid() << ",";
        }
        std::cout << "], suc=[";
        for (auto suc : op->getSuccessors()) {
            std::cout << suc->getGuid() << ",";
        }
        std::cout << "], outputs=[";
        for (auto output : op->getOutputs()) {
            std::cout << output->getHash()
                      << nnet::serializeVec(output->getDims()) << ",";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "    output: ";
    for (auto output : getOutputs()) {
        if (output->isNotCounted()) {
            std::cout << "(" << output->getHash() << ") ";
        } else {
            std::cout << output->getHash() << " ";
        }
    }
    std::cout << std::endl;
    return 0;
}

int SubGraph::printBrief() {
    std::cout << "Subgraph[" << getHash() << "] brief" << std::endl;
    std::cout << "    op num: " << getOperators().size() << std::endl;
    std::cout << "    input: ";
    for (auto input : getInputs()) {
        std::cout << input->getHash() << " ";
    }
    std::cout << std::endl;
    std::cout << "    output: ";
    for (auto output : getOutputs()) {
        if (output->isNotCounted()) {
            std::cout << "(" << output->getHash() << ") ";
        } else {
            std::cout << output->getHash() << " ";
        }
    }
    std::cout << std::endl;
    return 0;
}

int SubGraph::getComputeOps(std::vector<Operator *> &opList) {
    opList.clear();
    for (auto op : ops) {
        if (op->isComputeOp()) {
            opList.emplace_back(op);
        }
    }
    return 0;
};

int SubGraph::reset(std::vector<Tensor *> &input,
                    std::vector<Tensor *> &output) {
    auto &origin_input = getInputs();
    auto &origin_output = getOutputs();

    std::unordered_map<int, int> map;
    map.clear();
    for (auto t : tensors) {
        t->refresh();
    }

    if (origin_input.size() != input.size()) {
        return 1;
    }
    for (size_t i = 0, iEnd = input.size(); i < iEnd; i++) {
        origin_input[i]->replace(*input[i]);
    }
    if (origin_output.size() != output.size()) {
        return 1;
    }
    for (size_t i = 0, iEnd = output.size(); i < iEnd; i++) {
        origin_output[i]->replace(*output[i]);
    }

    return 0;
}

int SubGraph::split(std::shared_ptr<SubGraph> &master,
                    std::shared_ptr<SubGraph> &slave,
                    std::vector<Operator *> &slaveOps) {
    std::unordered_set<Operator *> set, slaveSet;
    for (auto op : getOperators()) {
        set.emplace(op);
    }
    for (auto op : slaveOps) {
        if (set.find(op) == set.end()) {
            std::cout << "ERROR: [SubGraph::split] slave op not in subgraph."
                      << std::endl;
            return 1;
        }
        slaveSet.emplace(op);
    }
    std::vector<Operator *> masterOps;
    for (auto op : getOperators()) {
        if (slaveSet.find(op) == slaveSet.end()) {
            masterOps.emplace_back(op);
        }
    }
    master = std::make_shared<SubGraph>(masterOps);
    slave = std::make_shared<SubGraph>(slaveOps);
    return 0;
}

int SubGraph::merge(std::shared_ptr<SubGraph> &master,
                    std::shared_ptr<SubGraph> &slave) {
    std::unordered_map<uint64_t, Tensor *> map;
    std::vector<Operator *> ops;
    for (auto t : getTensors()) {
        map.emplace(t->getHash(), t);
    }
    for (auto op : getOperators()) {
        ops.emplace_back(op);
    }
    for (auto op : slave->getOperators()) {
        std::vector<Tensor *> inputs;
        for (auto t : op->getInputs()) {
            if (map.find(t->getHash()) != map.end()) {
                inputs.emplace_back(map[t->getHash()]);
            } else {
                inputs.emplace_back(t);
            }
        }
        std::vector<Tensor *> outputs;
        for (auto t : op->getOutputs()) {
            if (map.find(t->getHash()) != map.end()) {
                outputs.emplace_back(map[t->getHash()]);
            } else {
                outputs.emplace_back(t);
            }
        }
        op->setInputs(inputs);
        op->setOutputs(outputs);
        ops.emplace_back(op);
    }
    master = std::make_shared<SubGraph>(ops);
    return 0;
}

bool Graph::mutateInceptionHead() {
    const int max_dep = 4;
    const std::vector<Operator::OpType> layer_ops{
        Operator::Gather, Operator::Reshape, Operator::Mul, Operator::Add};
    std::vector<std::vector<Tensor *>> layers_inputs(max_dep + 1);
    std::vector<Operator *> head_ops;

    Tensor *input = nullptr;
    for (auto t : tensors) {
        if (t->getType() == Tensor::Input) {
            input = t;
            break;
        }
    }
    if (input == nullptr)
        return false;
    layers_inputs[0].push_back(input);
    // TODO why there are more than 1 Input Tensor?

    for (int i = 0; i < max_dep; ++i) {
        int cnt = 0;
        for (auto input : layers_inputs[i]) {
            for (auto op : ops) {
                if (std::find(op->getInputs().begin(), op->getInputs().end(),
                              input) == op->getInputs().end())
                    continue;
                if (op->getType() != layer_ops[i] ||
                    op->getOutputs().size() != 1)
                    return false;
                cnt++;
                layers_inputs[i + 1].push_back(op->getOutputs()[0]);
                head_ops.push_back(op);
            }
        }
        if (cnt != 3)
            return false;
    }
    // This model is inception
    ConcatOp *concat_op = nullptr;
    ConvOp *conv_op = nullptr;
    for (auto op : ops) {
        if (std::find(op->getInputs().begin(), op->getInputs().end(),
                      layers_inputs[max_dep][0]) != op->getInputs().end())
            concat_op = dynamic_cast<ConcatOp *>(op);
    }
    for (auto op : ops) {
        if (std::find(op->getInputs().begin(), op->getInputs().end(),
                      concat_op->getOutputs()[0]) != op->getInputs().end())
            conv_op = dynamic_cast<ConvOp *>(op);
    }
    assert(concat_op && conv_op);
    head_ops.push_back(concat_op);
    Dim dim_weight{input->getDims()};
    auto mul_weight = tensor(dim_weight, Tensor::Float32);
    auto add_weight = tensor(dim_weight, Tensor::Float32);
    auto mul_op = this->mul({input, mul_weight});
    auto add_op = this->add({mul_op->getOutputs()[0], add_weight});
    conv_op->setInputs({add_op->getOutputs()[0], conv_op->getInputs()[1]});

    removeOps(head_ops);
    return true;
}

void GraphBase::removeOps(OpVec &removed_ops) {
    OpVec new_ops;
    TensorVec new_tensors, removed_tensors;
    for (Operator *op : ops) {
        bool removed = std::find(removed_ops.begin(), removed_ops.end(), op) !=
                       removed_ops.end();
        if (removed) {
            for (auto t : op->getOutputs()) {
                removed_tensors.push_back(t);
                delete t;
            }
            delete op;
        } else {
            new_ops.push_back(op);
        }
    }
    ops = new_ops;
    for (auto *t : tensors) {
        if (std::find(removed_tensors.begin(), removed_tensors.end(), t) !=
            removed_tensors.end())
            new_tensors.push_back(t);
    }
    tensors = new_tensors;
}

bool GraphBase::importOnnx(const char *net) {
    try {
        py::module::import("cpp_plugin").attr("import_onnx")(this, net);
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_ImportError)) {
            std::cerr << "Import Error. Don't forget to set environment "
                         "variable PYTHONPATH to contain "
                         "<repo-root>/python"
                      << std::endl;
        }
        throw;
    }

    updateConnection();
    return true;
}

bool GraphBase::exportOnnx(const char *path) {
    std::vector<std::string> tensor_name;
    std::map<std::string, std::string> tensor_dtype;
    std::map<std::string, std::vector<int>> tensor_dim;
    std::vector<std::string> initializer;
    std::vector<std::string> op_name;
    std::map<std::string, std::vector<std::string>> op_input, op_output;
    std::map<std::string, std::map<std::string, std::string>> op_attr;
    std::map<std::string, std::vector<int>> tensor_value;

    auto &tensors = getTensors();
    for (auto &tensor : tensors) {
        size_t guid = tensor->getGuid();
        Tensor::DataType dtype = tensor->getDType();
        Dim dim = tensor->getDims();
        Tensor::TensorType ttype = tensor->getType();

        std::string name = "tensor_" + std::to_string(guid);
        std::string dtype_str = "";
        if (dtype == Tensor::DataType::Float32) {
            dtype_str = "Float32";
        } else {
            dtype_str = "Int32";
        }
        tensor_name.emplace_back(name);
        tensor_dtype[name] = dtype_str;
        tensor_dim[name] = dim;

        if (ttype == Tensor::TensorType::Weight) {
            initializer.emplace_back(name);
        } else {
            if (ttype == Tensor::TensorType::Invalid) {
                std::cout << "Invalid tensor founded!" << std::endl;
            }
            if (ttype == Tensor::TensorType::NotCounted) {
                std::cout << "Notcounted tensor founded!" << std::endl;
            }
        }
    }

    std::cout << "Tensors Got." << std::endl;

    auto &operators = getOperators();
    for (auto &op : operators) {

        if (op->isReshapeOp()) {
            auto outp = op->getOutputs()[0];
            std::vector<int> extra;
            for (size_t i = 0, iEnd = (outp->getDims()).size(); i < iEnd; ++i) {
                extra.emplace_back((outp->getDims())[i]);
            }
            std::string opname =
                "Reshape_" + std::to_string(op->getGuid()) + "_shape";
            tensor_value[opname] = extra;
        }

        if (op->isTransposeOp()) {
            Tensor *inp = op->getInputs()[0];
            Tensor *oup = op->getOutputs()[0];
            std::string last = "tensor_" + std::to_string(inp->getGuid());
            Dim last_dim = inp->getDims();

            TransposeOp *trop = dynamic_cast<TransposeOp *>(op);
            const std::vector<std::shared_ptr<TransBasic>> &totop =
                trop->getTTParam();
            for (size_t i = 0, iEnd = totop.size(); i < iEnd; ++i) {
                std::string outp = "";
                if (i == iEnd - 1) {
                    outp = "tensor_" + std::to_string(oup->getGuid());
                } else {
                    outp = "tensor_tr_" + std::to_string(op->getGuid()) + "_" +
                           std::to_string(i);
                    tensor_name.emplace_back(outp);
                    tensor_dtype[outp] = tensor_dtype[last];
                }
                std::string optype = "";
                std::vector<int> extra;
                totop[i]->getOptypeDim(optype, last_dim, extra);

                if (i < iEnd - 1)
                    tensor_dim[outp] = last_dim;
                std::string opname = optype + "_tr_" +
                                     std::to_string(op->getGuid()) + "_" +
                                     std::to_string(i);
                op_name.emplace_back(opname);
                op_input[opname] = {last};
                op_output[opname] = {outp};
                last = outp;

                if (optype == "Reshape") {
                    tensor_value[opname + "_shape"] = extra;
                    std::map<std::string, std::string> attr;
                    op_attr[opname] = attr;
                }
                if (optype == "Transpose") {
                    std::map<std::string, std::string> attr;
                    std::string perm = "";
                    for (auto x : extra)
                        perm += std::to_string(x) + ",";
                    perm.pop_back();
                    attr["perm"] = "[" + perm + "]";
                    op_attr[opname] = attr;
                }
            }
            continue;
        }

        size_t guid = op->getGuid();
        std::vector<std::string> inp, oup;

        auto &inputtensors = op->getInputs();
        for (auto &tensor : inputtensors) {
            size_t guid = tensor->getGuid();
            std::string name = "tensor_" + std::to_string(guid);
            inp.emplace_back(name);
        }

        auto &outputtensors = op->getOutputs();
        for (auto &tensor : outputtensors) {
            size_t guid = tensor->getGuid();
            std::string name = "tensor_" + std::to_string(guid);
            oup.emplace_back(name);
        }

        std::string optype;
        std::map<std::string, std::string> attr;
        std::map<std::string, std::vector<int>> extra;
        op->getOptypeAttr(optype, attr, extra);

        for (auto it = extra.begin(); it != extra.end(); ++it) {
            std::string name = it->first;
            tensor_name.emplace_back(name);
            tensor_dtype[name] = "Float32";
            tensor_dim[name] = it->second;
            initializer.emplace_back(name);
            inp.emplace_back(name);
        }

        std::string opname = optype + "_" + std::to_string(guid);
        op_name.emplace_back(opname);
        op_input[opname] = inp;
        op_output[opname] = oup;
        op_attr[opname] = attr;
    }

    std::cout << "Operators Got." << std::endl;

    try {
        py::module::import("cpp_plugin")
            .attr("export_onnx")(path, tensor_name, tensor_dtype, tensor_dim,
                                 initializer, op_name, op_input, op_output,
                                 op_attr, tensor_value);
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_ImportError)) {
            std::cerr << "Import Error. Don't forget to set environment "
                         "variable PYTHONPATH to contain "
                         "<repo-root>/python"
                      << std::endl;
        }
        throw;
    }

    return true;
}

// compute subGraph in topo order
bool SubGraph::compute() {
    auto &opList = getOperators();
    std::vector<int> cnt(opList.size());
    std::unordered_map<int, int> opMap;
    std::vector<int> q;
    for (size_t i = 0, iEnd = opList.size(); i < iEnd; i++) {
        auto &op = opList[i];
        cnt[i] = op->getPredecessors().size();
        opMap.emplace(op->getGuid(), i);
        if (cnt[i] == 0) {
            q.emplace_back(i);
        }
    }
    int st = 0, ed = q.size();
    while (st < ed) {
        int id = q[st];
        st++;
        auto &op = opList[id];
        if (op->compute() == nullptr) {
            return false;
        }
        for (auto suc : op->getSuccessors()) {
            int suc_id = opMap[suc->getGuid()];
            cnt[suc_id]--;
            if (cnt[suc_id] == 0) {
                q.emplace_back(suc_id);
                ed++;
            }
        }
    }
    return true;
}

bool SubGraph::verification(SubGraph *mutant_graph, bool isFullComputing) {
    const int nSamples = 8;
    // Init inputs
    assert(inputs.size() == mutant_graph->getInputs().size());
    for (size_t i = 0, iEnd = inputs.size(); i < iEnd; i++) {
        auto input = inputs[i];
        auto input_mut = mutant_graph->getInputs()[i];
        assert(input->size() == input_mut->size());
        // Generate the same input data
        input->dataRand();
        input_mut->dataMalloc();
        input_mut->setData(input->getDataPtr());
        // // Generate debug data
        // auto input_d = input->getDataPtr();
        // auto input_mut_d = input_mut->getDataPtr();
        // for (size_t j = 0, jEnd = input->size(); j < jEnd; j++) {
        //     input_d[j] = input_mut_d[j] = 0;
        // }
    }
    // verify outputs
    bool ret = true;
    if (isFullComputing == true) {
        if (!compute() || !mutant_graph->compute()) {
            assert(!"compute failed");
            return false;
        }
        assert(inputs.size() == mutant_graph->getInputs().size());
        for (size_t i = 0, iEnd = outputs.size(); i < iEnd; i++) {
            auto output = outputs[i];
            auto output_mut = mutant_graph->getOutputs()[i];
            assert(output->size() == output_mut->size());
            auto output_d = output->getDataPtr();
            auto output_mut_d = output_mut->getDataPtr();
            for (size_t j = 0, jEnd = output->size(); j < jEnd; j++) {
                if (output_d[j] != output_mut_d[j]) {
                    dbg(j);
                    printf("[Wrong] graph = %d bestGraph = %d\n", output_d[j],
                           output_mut_d[j]);
                    return false;
                }
            }
        }
    } else {
        assert(false); // Data dependence of membound is not implemented
        std::vector<std::vector<std::pair<Dim, VType>>> computingPos;
        for (size_t i = 0; i < outputs.size(); i++) {
            // choose 8 random output positions
            computingPos.emplace_back();
            auto &back = computingPos.back();
            const auto dm = outputs[i]->getDims();
            srand(time(NULL));
            for (int j = 0; j < nSamples; j++) {
                Dim randPos = {};
                for (auto d : dm) {
                    int val = ((rand() % 2) + 1) * d / 3;
                    randPos.emplace_back(val);
                }
                back.emplace_back(std::make_pair(randPos, 0));
            }
            // compute and compare
            for (const auto &pos : back) {
                auto comp = compute(pos.first, i);
                auto comp_mut = mutant_graph->compute(pos.first, i);
                if (!comp.first || !comp_mut.first) {
                    printf("compute failed\n");
                    return false;
                }
                if (comp.second != comp_mut.second) {
                    dbg(pos);
                    printf("[Wrong] graph = %d bestGraph = %d\n", comp.second,
                           comp_mut.second);
                    ret = false;
                } else {
                    printf("graph = %d bestGraph = %d\n", comp.second,
                           comp_mut.second);
                }
            }
        }
    }
    return ret;
}

} // namespace tpm
