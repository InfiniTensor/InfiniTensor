#include "core/graph.h"
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

void GraphObj::optimize() {
    for (auto &op : ops) {
        switch (op->getOpType()) {
        default:
            break;
        }
    }
}

void GraphObj::dataMalloc() {
    for (auto &tensor : tensors) {
        if (tensor->getSource() && tensor->getTargets().size() > 0 &&
            tensor->getSource()->getOpType() == OpType::Reshape) {
            continue;
        } else
            tensor->dataMalloc();
    }
    // Fill reshape output for avoiding nullptr
    for (auto &tensor : tensors) {
        if (tensor->getSource() &&
            tensor->getSource()->getOpType() == OpType::Reshape) {
            tensor->setData(tensor->getSource()->getInputs(0)->getDataBlob());
        }
    }
}

void GraphObj::dataFree() {
    for (auto &tensor : tensors) {
        tensor->freeData();
    }
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype, TensorType tensorType) {
    return tensors.emplace_back(
        make_ref<TensorObj>(dim, dtype, runtime, tensorType));
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
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
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

int GraphObj::removeIndependentTensors() {
    TensorVec newTensors;
    for (const auto &t : tensors)
        if (!t->getTargets().empty() || t->getSource())
            newTensors.emplace_back(t);
    auto ret = tensors.size() - newTensors.size();
    tensors = newTensors;
    return ret;
}

} // namespace infini
