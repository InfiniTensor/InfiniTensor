#include "core/graph.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce_mean.h"
#include "operators/unary.h"
#include "optimization/common.h"
#include <algorithm>
#include <queue>

namespace infini {

GraphObj::GraphObj(Runtime runtime, OpVec ops_in)
    : runtime(runtime), sorted(false) {
    map<UidBaseType, Tensor> tensorPool;
    // Clone tensors
    for (const auto &op : ops_in) {
        for (const auto &t : op->getInputs())
            if (tensorPool.find(t->getFuid()) == tensorPool.end())
                tensorPool[t->getFuid()] = cloneTensor(t);
        for (const auto &t : op->getOutputs())
            if (tensorPool.find(t->getFuid()) == tensorPool.end())
                tensorPool[t->getFuid()] = cloneTensor(t);
    }
    // Clone operators and add connections
    for (const auto &op : ops_in) {
        TensorVec inputs, outputs;
        for (const auto &t : op->getInputs())
            inputs.emplace_back(tensorPool.at(t->getFuid()));
        for (const auto &t : op->getOutputs())
            outputs.emplace_back(tensorPool.at(t->getFuid()));
        addOperatorAndConnect(op->clone(inputs, outputs));
    }
}

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        input->addTarget(op);
        if (auto pred = input->getSource()) {
            pred->addSuccessors(op);
            op->addPredecessors(pred);
        }
    }
    for (auto &output : op->getOutputs()) {
        output->setSource(op);
        for (auto &succ : output->getTargets()) {
            succ->addPredecessors(op);
            op->addSuccessors(succ);
        }
    }
}

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

bool GraphObj::topo_sort() {
    if (this->sorted)
        return true;

    // std::unordered_set<Tensor> inputs;
    std::unordered_set<Operator> waiting(this->ops.begin(), this->ops.end());
    std::vector<Operator> sorted;

    while (!waiting.empty()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        // Find head nodes.
        for (auto it = waiting.begin(); it != waiting.end();) {
            const auto &this_inputs = (*it)->getInputs();
            // If none of the input tensors is in waiting list,
            // this node is a head node.
            const auto is_head = std::all_of(
                this_inputs.begin(), this_inputs.end(), [&](const auto &input) {
                    auto src = input->getSource();
                    return src // If the source node is in the waiting list,
                               // means that this node is not the head node.
                               ? waiting.find(src) == waiting.end()
                               // This tensor has no source node,
                               // it must be a input tensor.
                               : (/*inputs.insert(input),*/ true);
                });
            // Moves head node to sorted.
            if (is_head) {
                modified = true;
                sorted.emplace_back(std::move(*it));
                it = waiting.erase(it);
            } else {
                ++it;
            }
        }
        // Waiting list never modifies during a pass,
        // sorting fails.
        if (!modified) {
            return false;
        }
    }

    // Done.
    this->ops = std::move(sorted);
    return this->sorted = true;
}

optimization::DataType cast(DataType ty) {
#define IT(A, B)                                                               \
    if (ty == DataType::A)                                                     \
        return {optimization::DataTypeId::B};

    IT(Float32, FLOAT)          //
    else IT(UInt32, UINT32)     //
        else IT(UInt8, UINT8)   //
        else IT(Int8, INT8)     //
        else IT(UInt16, UINT16) //
        else IT(Int16, INT16)   //
        else IT(Int32, INT32)   //
        else IT(Int64, INT64)   //
        else IT_ASSERT(false, "unsupported data type");

#undef IT
}

void GraphObj::optimize() {
    namespace opt = optimization;

    topo_sort();

#define I(PTR) reinterpret_cast<uintptr_t>((PTR).get())

    unordered_map<uintptr_t, opt::Arc<opt::Tensor>> tensors;
    for (const auto &t : this->getTensors()) {
        const auto dims = t->getDims();
        opt::Vec<size_t> shape(dims.size());
        std::transform(dims.begin(), dims.end(), shape.begin(),
                       [](auto x) { return static_cast<size_t>(x); });

        opt::Data data{};
        if (t->hasData()) {
            auto origin = t->getDataBlob();
            data.cpu_data.resize(t->getBytes());
            memcpy(data.cpu_data.data(), origin->getPtr<uint8_t *>(),
                   data.cpu_data.size());
        }
        tensors[I(t)] =
            opt::Tensor::share(shape, cast(t->getDType()), std::move(data));
    }

    opt::Unigraph ans;

    for (const auto &op : this->getOperators()) {
        const auto inputs = op->getInputs(), outputs = op->getOutputs();
        opt::Vec<opt::Arc<opt::Tensor>> in(inputs.size()), out(outputs.size());
        std::transform(inputs.begin(), inputs.end(), in.begin(),
                       [&](auto x) { return tensors[I(x)]; });
        std::transform(outputs.begin(), outputs.end(), out.begin(),
                       [&](auto x) { return tensors[I(x)]; });
        switch (op->getOpType()) {
        case OpType::Abs:
            ans.push_operator(opt::OpType::Abs, std::move(in), std::move(out));
            break;
        case OpType::ACos:
            ans.push_operator(opt::OpType::Acos, std::move(in), std::move(out));
            break;
        case OpType::ACosH:
            ans.push_operator(opt::OpType::Acosh, std::move(in),
                              std::move(out));
            break;
        case OpType::Add:
            ans.push_operator(opt::OpType::Add, std::move(in), std::move(out));
            break;
        case OpType::And:
            ans.push_operator(opt::OpType::And, std::move(in), std::move(out));
            break;
        case OpType::ASin:
            ans.push_operator(opt::OpType::Asin, std::move(in), std::move(out));
            break;
        case OpType::ASinH:
            ans.push_operator(opt::OpType::Asinh, std::move(in),
                              std::move(out));
            break;
        case OpType::ATan:
            ans.push_operator(opt::OpType::Atan, std::move(in), std::move(out));
            break;
        case OpType::ATanH:
            ans.push_operator(opt::OpType::Atanh, std::move(in),
                              std::move(out));
            break;
        case OpType::AvgPool: {
            auto obj = as<AvgPoolObj>(op);
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getDh(), obj->getDw()}));
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getKh(), obj->getKw()}));
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getPh(), obj->getPw()}));
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getSh(), obj->getSw()}));
            ans.push_operator(opt::OpType::AveragePool, std::move(in),
                              std::move(out));
        } break;
        case OpType::BatchNorm:
            ans.push_operator(opt::OpType::BatchNormalization, std::move(in),
                              std::move(out));
            break;
        case OpType::BitLeftShift:
            in.push_back(opt::Tensor::share_single<uint8_t>(0));
            ans.push_operator(opt::OpType::BitShift, std::move(in),
                              std::move(out));
            break;
        case OpType::BitRightShift:
            in.push_back(opt::Tensor::share_single<uint8_t>(1));
            ans.push_operator(opt::OpType::BitShift, std::move(in),
                              std::move(out));
            break;
        case OpType::BitAnd:
            ans.push_operator(opt::OpType::BitwiseAnd, std::move(in),
                              std::move(out));
            break;
        case OpType::BitNot:
            ans.push_operator(opt::OpType::BitwiseNot, std::move(in),
                              std::move(out));
            break;
        case OpType::BitOr:
            ans.push_operator(opt::OpType::BitwiseOr, std::move(in),
                              std::move(out));
            break;
        case OpType::BitXor:
            ans.push_operator(opt::OpType::BitwiseXor, std::move(in),
                              std::move(out));
            break;
        case OpType::Cast:
            ans.push_operator(opt::OpType::Cast, std::move(in), std::move(out));
            break;
        case OpType::Ceil:
            ans.push_operator(opt::OpType::Ceil, std::move(in), std::move(out));
            break;
        case OpType::Clip: {
            auto obj = as<ClipObj>(op);
            auto min = obj->getMin();
            auto max = obj->getMax();
            in.push_back(
                opt::Tensor::share_single<float>(min ? *min : -INFINITY));
            in.push_back(
                opt::Tensor::share_single<float>(max ? *max : INFINITY));
            ans.push_operator(opt::OpType::Clip, std::move(in), std::move(out));
        } break;
        case OpType::Concat:
            in.push_back(
                opt::Tensor::share_single<int>(as<ConcatObj>(op)->getDim()));
            ans.push_operator(opt::OpType::Concat, std::move(in),
                              std::move(out));
            break;
        case OpType::Conv: {
            auto obj = as<ConvObj>(op);
            in.push_back(opt::Tensor::share_vec<size_t>(
                {(size_t)obj->getDh(), (size_t)obj->getDw()}));
            in.push_back(opt::Tensor::share_vec<size_t>(
                {(size_t)obj->getPh(), (size_t)obj->getPw()}));
            in.push_back(opt::Tensor::share_vec<size_t>(
                {(size_t)obj->getSh(), (size_t)obj->getSw()}));
            ans.push_operator(opt::OpType::Conv, std::move(in), std::move(out));
        } break;
        case OpType::Cos:
            ans.push_operator(opt::OpType::Cos, std::move(in), std::move(out));
            break;
        case OpType::CosH:
            ans.push_operator(opt::OpType::Cosh, std::move(in), std::move(out));
            break;
        case OpType::Div:
            ans.push_operator(opt::OpType::Div, std::move(in), std::move(out));
            break;
        case OpType::Dropout:
            ans.push_operator(opt::OpType::Dropout, std::move(in),
                              std::move(out));
            break;
        case OpType::Exp:
            ans.push_operator(opt::OpType::Exp, std::move(in), std::move(out));
            break;
        case OpType::Flatten:
            ans.push_operator(opt::OpType::Flatten, std::move(in),
                              std::move(out));
            break;
        case OpType::Floor:
            ans.push_operator(opt::OpType::Floor, std::move(in),
                              std::move(out));
            break;
        case OpType::Gather:
            in.push_back(
                opt::Tensor::share_single<int>(as<GatherObj>(op)->getAxis()));
            ans.push_operator(opt::OpType::Gather, std::move(in),
                              std::move(out));
            break;
        case OpType::GreaterThan:
            ans.push_operator(opt::OpType::Greater, std::move(in),
                              std::move(out));
            break;
        case OpType::GreaterEqual:
            ans.push_operator(opt::OpType::GreaterOrEqual, std::move(in),
                              std::move(out));
            break;
        case OpType::Identity:
            ans.push_operator(opt::OpType::Identity, std::move(in),
                              std::move(out));
            break;
        case OpType::Log:
            ans.push_operator(opt::OpType::Log, std::move(in), std::move(out));
            break;
        case OpType::Matmul: {
            auto obj = as<MatmulObj>(op);
            IT_ASSERT(obj->getAct() == ActType::None);
            in.push_back(opt::Tensor::share_single<float>(1.0f));
            in.push_back(opt::Tensor::share_single<float>(1.0f));
            in.push_back(
                opt::Tensor::share_single<int>(obj->getTransA() ? 1 : 0));
            in.push_back(
                opt::Tensor::share_single<int>(obj->getTransB() ? 1 : 0));
            ans.push_operator(opt::OpType::Gemm, std::move(in), std::move(out));
        } break;
        case OpType::Maximum:
            ans.push_operator(opt::OpType::Max, std::move(in), std::move(out));
            break;
        case OpType::MaxPool: {
            auto obj = as<MaxPoolObj>(op);
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getDh(), obj->getDw()}));
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getKh(), obj->getKw()}));
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getPh(), obj->getPw()}));
            in.push_back(
                opt::Tensor::share_vec<int>({obj->getSh(), obj->getSw()}));
            ans.push_operator(opt::OpType::AveragePool, std::move(in),
                              std::move(out));
        } break;
        case OpType::Minimum:
            ans.push_operator(opt::OpType::Min, std::move(in), std::move(out));
            break;
        case OpType::Mul:
            ans.push_operator(opt::OpType::Mul, std::move(in), std::move(out));
            break;
        case OpType::Neg:
            ans.push_operator(opt::OpType::Neg, std::move(in), std::move(out));
            break;
        case OpType::Not:
            ans.push_operator(opt::OpType::Not, std::move(in), std::move(out));
            break;
        case OpType::Or:
            ans.push_operator(opt::OpType::Or, std::move(in), std::move(out));
            break;
        case OpType::Pad:
            in.push_back(
                opt::Tensor::share_vec<int>(as<PadObj>(op)->getPads()));
            ans.push_operator(opt::OpType::Pad, std::move(in), std::move(out));
            break;
        case OpType::Reciprocal:
            ans.push_operator(opt::OpType::Reciprocal, std::move(in),
                              std::move(out));
            break;
        case OpType::ReduceMean: {
            const auto obj = as<ReduceMeanObj>(op);
            const auto axes = obj->getAxes();
            in.push_back(
                opt::Tensor::share_vec<int>(vector(axes.begin(), axes.end())));
            in.push_back(
                opt::Tensor::share_single<int>(obj->getKeepDims() ? 1 : 0));
            ans.push_operator(opt::OpType::ReduceMean, std::move(in),
                              std::move(out));
        } break;
        case OpType::Relu:
            ans.push_operator(opt::OpType::Relu, std::move(in), std::move(out));
            break;
        case OpType::Reshape:
            ans.push_operator(opt::OpType::Reshape, std::move(in),
                              std::move(out));
            break;
        case OpType::Resize:
            ans.push_operator(opt::OpType::Resize, std::move(in),
                              std::move(out));
            break;
        case OpType::Round:
            ans.push_operator(opt::OpType::Round, std::move(in),
                              std::move(out));
            break;
        case OpType::Shape:
            ans.push_operator(opt::OpType::Shape, std::move(in),
                              std::move(out));
            break;
        case OpType::Sigmoid:
            ans.push_operator(opt::OpType::Sigmoid, std::move(in),
                              std::move(out));
            break;
        case OpType::Sin:
            ans.push_operator(opt::OpType::Sin, std::move(in), std::move(out));
            break;
        case OpType::SinH:
            ans.push_operator(opt::OpType::Sinh, std::move(in), std::move(out));
            break;
        case OpType::Slice:
            IT_TODO_HALT();
            ans.push_operator(opt::OpType::Slice, std::move(in),
                              std::move(out));
            break;
        case OpType::Softmax:
            ans.push_operator(opt::OpType::Softmax, std::move(in),
                              std::move(out));
            break;
        case OpType::Split:
            ans.push_operator(opt::OpType::Split, std::move(in),
                              std::move(out));
            break;
        case OpType::Sqrt:
            ans.push_operator(opt::OpType::Sqrt, std::move(in), std::move(out));
            break;
        case OpType::Sub:
            ans.push_operator(opt::OpType::Sub, std::move(in), std::move(out));
            break;
        case OpType::Tan:
            ans.push_operator(opt::OpType::Tan, std::move(in), std::move(out));
            break;
        case OpType::TanH:
            ans.push_operator(opt::OpType::Tanh, std::move(in), std::move(out));
            break;
        case OpType::Transpose:
            ans.push_operator(opt::OpType::Tanh, std::move(in), std::move(out));
            break;
        case OpType::Xor:
            ans.push_operator(opt::OpType::Xor, std::move(in), std::move(out));
            break;
        default:
            break;
        }
    }

#undef I
}

void GraphObj::dataMalloc() {
    for (auto &tensor : tensors) {
        tensor->dataMalloc();
    }
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

OpVec GraphObj::getComputeOps() const {
    OpVec opList;
    for (auto op : ops)
        if (op->isComputeOp())
            opList.emplace_back(op);
    return opList;
}

void GraphObj::deleteConnection(Tensor tensor, Operator op) {
    // if op is target
    IT_ASSERT(std::find(tensor->getTargets().begin(),
                        tensor->getTargets().end(),
                        op) != tensor->getTargets().end());
    tensor->removeTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->removeSuccessors(op);
        op->removePredecessors(tensor->getSource());
    }
}

// add op as a target
void GraphObj::addConnection(Tensor tensor, Operator op) {
    tensor->addTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->addSuccessors(op);
        op->addPredecessors(tensor->getSource());
    }
}

void GraphObj::replaceConnection(Tensor oldTensor, Tensor newTensor,
                                 Operator op) {
    // op is a target of old tensor
    IT_ASSERT(std::find(oldTensor->getTargets().begin(),
                        oldTensor->getTargets().end(),
                        op) != oldTensor->getTargets().end());
    addConnection(newTensor, op);
    deleteConnection(oldTensor, op);
    op->replaceInput(oldTensor, newTensor);
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in
// "ops".
bool GraphObj::checkValid() const {
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                    nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                      tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini
