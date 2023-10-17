#include "core/graph.h"
#include <algorithm>
#include <queue>

namespace infini {

GraphObj::GraphObj(Runtime runtime, OpVec ops_in)
    : runtime(runtime), allocator(runtime), sorted(false) {
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
        switch (op->getOpType().underlying()) {
        default:
            break;
        }
    }
}

void GraphObj::dataMalloc(bool useNaiveAllocator, size_t memPoolSize) {
    // topological sorting first
    IT_ASSERT(topo_sort() == true);
    if (useNaiveAllocator) {
        // can not set memory pool when use naive allocator
        IT_ASSERT(memPoolSize == 0);
        // used for debugging memory out-of-bounds access, tensors will not be
        // released correctly
        // note: behavior may not match running in non-naive mode, and it may
        // not reproduce the bug
        for (auto &tensor : tensors) {
            tensor->dataMalloc();
        }
        return;
    }
    if (memPoolSize > 0) {
        allocator.setMemPool(memPoolSize);
    }
    // count the number of times all tensors are used
    std::unordered_map<TensorObj *, size_t> tensorToRefCount;
    // record the memory address offsets of all tensors to be allocated
    std::unordered_map<TensorObj *, size_t> tensorToOffset;

    // reinit allocator
    allocator.init();

    // record all weight tensors, including weight tensors and kvcache
    // tensors
    std::unordered_set<TensorObj *> weightTensors;
    for (auto &tensor : tensors) {
        if (tensor->isWeight()) {
            // allocate memory for all weight tensors first, and this memory
            // will not be freed until the graph is destroyed
            weightTensors.insert(tensor.get());
            if (!this->weightAllocated) {
                tensorToOffset[tensor.get()] =
                    allocator.allocWeight(tensor->getBytes());
            }
        } else if (tensor->isInput() || tensor->isOutput()) {
            // allocate memory for all input and output tensors, and this memory
            // will not be reused later
            tensorToOffset[tensor.get()] = allocator.alloc(tensor->getBytes());
        } else {
            tensorToRefCount[tensor.get()] = tensor->getTargets().size();
            // allocate memory for all user-created tensors
            if (tensor.get()->getSource() == nullptr) {
                tensorToOffset[tensor.get()] =
                    allocator.alloc(tensor->getBytes());
            }
        }
    }
    // if memory has not yet been allocated for weight tensors,
    // allocate memory now and do not allocate again in the future.
    if (!this->weightAllocated) {
        this->weightAllocated = true;
        // only allocate once for weight tensors
        for (auto &tensor : weightTensors) {
            IT_ASSERT(tensorToOffset.find(tensor) != tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getWeightPtr()) +
                    tensorToOffset[tensor]));
        }
    }
    // traverse in topological order and simulate memory allocation
    for (auto &op : ops) {
        // memory should be allocated for the op's output first
        auto outputs = op->getOutputs();
        for (auto &tensor : outputs) {
            if (tensor->isOthers()) {
                tensorToOffset[tensor.get()] =
                    allocator.alloc(tensor->getBytes());
            }
        }
        auto inputs = op->getInputs();
        for (auto &tensor : inputs) {
            if (tensor->isOthers()) {
                auto tensorIter = tensorToRefCount.find(tensor.get());
                IT_ASSERT(tensorIter != tensorToRefCount.end());
                IT_ASSERT(tensorToRefCount[tensor.get()] > 0);
                tensorToRefCount[tensor.get()] -= 1;
                if (tensorToRefCount[tensor.get()] == 0) {
                    // indicate that this tensor will no longer be used and
                    // perform memory free
                    tensorToRefCount.erase(tensor.get());
                    allocator.free(tensorToOffset[tensor.get()],
                                   tensor->getBytes());
                }
            }
        }
    }

    // perform actual memory allocation for non-weight tensors
    for (auto &tensor : tensors) {
        if (!tensor->isWeight()) {
            IT_ASSERT(tensorToOffset.find(tensor.get()) !=
                      tensorToOffset.end());
            tensor->setDataBlob(make_ref<BlobObj>(
                tensor->runtime, static_cast<uint8_t *>(allocator.getPtr()) +
                                     tensorToOffset[tensor.get()]));
        }
    }
}

Tensor GraphObj::cloneKV(Tensor &tensor) {
    auto obj = tensor->clone();
    if (allocator.getMemPoolStatus()) {
        if (tensor->hasData()) {
            obj->setDataBlob(make_ref<BlobObj>(
                tensor->runtime,
                static_cast<uint8_t *>(allocator.getHeapPtr()) +
                    allocator.heapAlloc(tensor->getBytes())));
            obj->copyData(tensor);
        }
    } else {
        if (tensor->hasData()) {
            obj->dataMalloc();
            obj->copyData(tensor);
        }
    }
    return obj;
}

void GraphObj::freeHeap() { this->allocator.freeHeap(); }

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
        if (op->getOpType().isMatMulOrConv())
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

} // namespace infini
