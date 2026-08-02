#include "core/graph.h"
#include "operators/reshape.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini {

GraphObj::GraphObj(Runtime runtime, OpVec ops_in)
    : runtime(runtime), allocator(runtime), sorted(false) {
    map<UidBaseType, Tensor> tensorPool;
    // Clone tensors
    for (const auto &op : ops_in) {
        for (const auto &t : op->getInputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
        for (const auto &t : op->getOutputs()) {
            if (t) {
                if (tensorPool.find(t->getFuid()) == tensorPool.end())
                    tensorPool[t->getFuid()] = cloneTensor(t);
            }
        }
    }
    // Clone operators and add connections
    for (const auto &op : ops_in) {
        TensorVec inputs, outputs;
        for (const auto &t : op->getInputs()) {
            if (t) {
                inputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }

        for (const auto &t : op->getOutputs()) {
            if (t) {
                outputs.emplace_back(tensorPool.at(t->getFuid()));
            }
        }
        addOperatorAndConnect(op->clone(inputs, outputs));
    }
}

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
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
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted;
    std::unordered_set<OperatorObj *> flags;
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const &op : ops) {
            if (auto const &inputs = op->getInputs();
                flags.find(op.get()) == flags.end() &&
                std::all_of(inputs.begin(), inputs.end(),
                            [&flags](auto const &input) {
                                auto ptr = input->getSource().get();
                                return !ptr || flags.find(ptr) != flags.end();
                            })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) {
            return false;
        }
    }
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

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}

void GraphObj::lockAllocationMode(bool useNaiveAllocator, size_t memPoolSize) {
    AllocationMode requestedMode;
    if (useNaiveAllocator) {
        IT_ASSERT(memPoolSize == 0,
                  "Naive allocator cannot use a fixed memory pool");
        requestedMode = AllocationMode::Naive;
    } else if (memPoolSize > 0 || allocationMode == AllocationMode::FixedPool) {
        requestedMode = AllocationMode::FixedPool;
    } else {
        requestedMode = AllocationMode::DynamicPool;
    }

    if (allocationMode == AllocationMode::Uninitialized) {
        allocationMode = requestedMode;
        if (requestedMode == AllocationMode::FixedPool)
            fixedPoolSize = memPoolSize;
        return;
    }

    IT_ASSERT(allocationMode == requestedMode,
              "Cannot change allocator mode after the first allocation");
    if (requestedMode == AllocationMode::FixedPool && memPoolSize > 0) {
        IT_ASSERT(memPoolSize == fixedPoolSize,
                  "Cannot change fixed memory pool size after allocation");
    }
}

void GraphObj::dataMalloc(bool useNaiveAllocator, size_t memPoolSize) {
    dataMallocImpl(useNaiveAllocator, memPoolSize, false);
}

void GraphObj::trimMemory() {
    IT_ASSERT(allocationMode == AllocationMode::DynamicPool,
              "trimMemory requires an allocated dynamic memory pool");
    IT_ASSERT(!allocator.hasLiveHeapBlobs(),
              "Cannot trim memory while heap tensors are still alive");
    dataMallocImpl(false, 0, true);
}

void GraphObj::dataMallocImpl(bool useNaiveAllocator, size_t memPoolSize,
                              bool trim) {
    if (allocationMode != AllocationMode::Uninitialized) {
        dataMallocImplCore(useNaiveAllocator, memPoolSize, trim);
        return;
    }

    vector<Blob> previousData;
    previousData.reserve(tensors.size());
    for (const auto &tensor : tensors)
        previousData.emplace_back(tensor->getDataBlob());

    const auto previousGeneration = allocationGeneration;
    try {
        dataMallocImplCore(useNaiveAllocator, memPoolSize, trim);
    } catch (...) {
        for (size_t i = 0; i < tensors.size(); ++i)
            tensors[i]->setDataBlob(previousData[i]);
        allocator.reset();
        allocationMode = AllocationMode::Uninitialized;
        fixedPoolSize = 0;
        weightAllocated = false;
        fixedPoolLayoutCommitted = false;
        fixedPoolTensorLayout.clear();
        fixedPoolActivationLayout.clear();
        allocationGeneration = previousGeneration;
        throw;
    }
}

void GraphObj::dataMallocImplCore(bool useNaiveAllocator, size_t memPoolSize,
                                  bool trim) {
    // topological sorting first

    IT_ASSERT(topo_sort() == true);
    lockAllocationMode(useNaiveAllocator, memPoolSize);

    using TensorMemoryState = std::pair<const void *, size_t>;
    const auto getTensorMemoryState = [](const Tensor &tensor) {
        if (!tensor->hasData())
            return TensorMemoryState{nullptr, 0};
        return TensorMemoryState{tensor->getRawDataPtr<const void *>(),
                                 tensor->getDataBlob()->getBytes()};
    };
    vector<TensorMemoryState> previousMemoryStates;
    previousMemoryStates.reserve(tensors.size());
    for (const auto &tensor : tensors)
        previousMemoryStates.emplace_back(getTensorMemoryState(tensor));
    const auto updateAllocationGeneration = [&]() {
        bool changed = previousMemoryStates.size() != tensors.size();
        for (size_t i = 0; !changed && i < tensors.size(); ++i) {
            if (previousMemoryStates[i] != getTensorMemoryState(tensors[i])) {
                changed = true;
            }
        }
        if (changed)
            ++allocationGeneration;
    };

    if (useNaiveAllocator) {
        // can not set memory pool when use naive allocator
        IT_ASSERT(memPoolSize == 0);
        // Used for debugging memory out-of-bounds access. Tensor memory is not
        // reused, so behavior may not match non-naive mode or reproduce the
        // same bug.
        for (auto &tensor : tensors) {
            if (!tensor->isWeight() ||
                (tensor->isWeight() && !weightAllocated)) {
                tensor->dataMalloc();
            }
        }
        weightAllocated = true;
        updateAllocationGeneration();
        return;
    }
    if (allocationMode == AllocationMode::FixedPool) {
        allocator.setMemPool(fixedPoolSize);
    }
    const bool hasFixedMemPool = allocator.getMemPoolStatus();
    // count the number of times all tensors are used
    std::unordered_map<TensorObj *, size_t> tensorToRefCount;
    // record the memory address offsets of all tensors to be allocated
    std::unordered_map<TensorObj *, size_t> tensorToOffset;

    // reinit allocator
    allocator.init();
    if (!weightAllocated)
        allocator.resetWeightPlan();

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
    const auto preserveData = [](TensorObj *tensor, const Blob &blob) {
        if (tensor->hasData() && tensor->getBytes() > 0 &&
            tensor->getDataBlob()->getBytes() == tensor->getBytes() &&
            tensor->getDataBlob()->getPtr<void *>() != blob->getPtr<void *>()) {
            auto copy = tensor->clone();
            copy->setDataBlob(blob);
            copy->copyData(tensor);
        }
    };
    // if memory has not yet been allocated for weight tensors,
    // allocate memory now and do not allocate again in the future.
    if (!this->weightAllocated) {
        vector<std::pair<TensorObj *, Blob>> weightBlobs;
        weightBlobs.reserve(weightTensors.size());
        for (auto &tensor : weightTensors) {
            IT_ASSERT(tensorToOffset.find(tensor) != tensorToOffset.end());
            if (tensor->hasData() &&
                tensor->getDataBlob()->getBytes() != tensor->getBytes())
                tensor->freeData();
            auto blob = allocator.getWeightBlob(tensorToOffset[tensor],
                                                tensor->getBytes());
            preserveData(tensor, blob);
            weightBlobs.emplace_back(tensor, std::move(blob));
        }
        for (const auto &[tensor, blob] : weightBlobs)
            tensor->setDataBlob(blob);
        this->weightAllocated = true;
    }
    // traverse in topological order and simulate memory allocation
    for (auto &op : ops) {
        // memory should be allocated for the op's output first
        auto outputs = op->getOutputs();
        for (auto &tensor : outputs) {
            if (tensor) {
                if (tensor->isOthers()) {
                    tensorToOffset[tensor.get()] =
                        allocator.alloc(tensor->getBytes());
                }
            }
        }
        auto inputs = op->getInputs();
        for (auto &tensor : inputs) {
            if (tensor) {
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
    }

    vector<std::pair<TensorObj *, size_t>> plannedTensorLayout;
    vector<std::pair<TensorObj *, size_t>> plannedActivationLayout;
    if (hasFixedMemPool) {
        plannedTensorLayout.reserve(tensors.size());
        plannedActivationLayout.reserve(tensors.size() - weightTensors.size());
        for (const auto &tensor : tensors) {
            plannedTensorLayout.emplace_back(tensor.get(), tensor->getBytes());
            if (!tensor->isWeight()) {
                const auto offset = tensorToOffset.find(tensor.get());
                IT_ASSERT(offset != tensorToOffset.end());
                plannedActivationLayout.emplace_back(tensor.get(),
                                                     offset->second);
            }
        }
        if (fixedPoolLayoutCommitted) {
            const bool layoutUnchanged =
                plannedTensorLayout == fixedPoolTensorLayout &&
                plannedActivationLayout == fixedPoolActivationLayout;
            IT_ASSERT(layoutUnchanged,
                      "Fixed memory pool does not support dynamic memory "
                      "layout changes");
        }
    }

    const auto clearInvalidActivationData = [&]() {
        for (auto &tensor : tensors) {
            if (!tensor->isWeight() && tensor->hasData() &&
                tensor->getDataBlob()->getBytes() != tensor->getBytes())
                tensor->freeData();
        }
    };

    Blob activationStorage;
    try {
        activationStorage = allocator.prepareActivationStorage(trim);
    } catch (...) {
        // Never leave an undersized view available to a later kernel launch.
        clearInvalidActivationData();
        throw;
    }
    clearInvalidActivationData();

    using TensorBlobPair = std::pair<TensorObj *, Blob>;
    const auto prepareActivationBlobs = [&](const Blob &storage) {
        vector<TensorBlobPair> blobs;
        blobs.reserve(tensors.size() - weightTensors.size());
        bool movesPreservedData = false;
        for (auto &tensor : tensors) {
            if (tensor->isWeight())
                continue;
            const auto offset = tensorToOffset.find(tensor.get());
            IT_ASSERT(offset != tensorToOffset.end());
            auto blob = allocator.getActivationBlob(storage, offset->second,
                                                    tensor->getBytes());
            if (tensor->getSource() == nullptr && tensor->hasData() &&
                tensor->getDataBlob()->getBytes() == tensor->getBytes() &&
                tensor->getDataBlob()->getPtr<void *>() !=
                    blob->getPtr<void *>()) {
                movesPreservedData = true;
            }
            blobs.emplace_back(tensor.get(), std::move(blob));
        }
        return std::make_pair(std::move(blobs), movesPreservedData);
    };

    auto [activationBlobs, movesPreservedData] =
        prepareActivationBlobs(activationStorage);
    if (!hasFixedMemPool && movesPreservedData &&
        allocator.isCurrentActivationStorage(activationStorage)) {
        // Moving live inputs inside the same pool can overwrite another input
        // before it is copied. Use a separate candidate storage in this rare
        // layout-changing case and preserve the transaction boundary.
        activationBlobs.clear();
        activationStorage = allocator.prepareActivationStorage(trim, true);
        activationBlobs = prepareActivationBlobs(activationStorage).first;
    }

    for (const auto &[tensor, blob] : activationBlobs) {
        if (tensor->getSource() == nullptr)
            preserveData(tensor, blob);
    }

    allocator.commitActivationStorage(activationStorage);
    if (hasFixedMemPool && !fixedPoolLayoutCommitted) {
        fixedPoolTensorLayout = std::move(plannedTensorLayout);
        fixedPoolActivationLayout = std::move(plannedActivationLayout);
        fixedPoolLayoutCommitted = true;
    }
    for (const auto &[tensor, blob] : activationBlobs)
        tensor->setDataBlob(blob);
    updateAllocationGeneration();
}

Tensor GraphObj::cloneKV(Tensor &tensor) {
    auto obj = tensor->clone();
    if (allocator.getMemPoolStatus()) {
        if (tensor->hasData()) {
            const auto previousHeapPeak = allocator.getHeapPeak();
            try {
                auto offset = allocator.heapAlloc(tensor->getBytes());
                obj->setDataBlob(
                    allocator.getHeapBlob(offset, tensor->getBytes()));
                obj->copyData(tensor);
            } catch (...) {
                obj->freeData();
                allocator.rollbackHeap(previousHeapPeak);
                throw;
            }
            ++allocationGeneration;
        }
    } else {
        if (tensor->hasData()) {
            obj->dataMalloc();
            obj->copyData(tensor);
        }
    }
    return obj;
}

void GraphObj::validateMemory() const {
    for (const auto &tensor : tensors) {
        IT_ASSERT(tensor != nullptr, "Graph contains a null tensor");
        IT_ASSERT(tensor->hasData(), "Tensor " +
                                         std::to_string(tensor->getFuid()) +
                                         " has no allocated memory");
        auto blob = tensor->getDataBlob();
        IT_ASSERT(blob->getBytes() >= tensor->getBytes(),
                  "Tensor " + std::to_string(tensor->getFuid()) + " requires " +
                      std::to_string(tensor->getBytes()) +
                      " bytes, but its Blob has only " +
                      std::to_string(blob->getBytes()));
        IT_ASSERT(blob->getPtr<void *>() != nullptr,
                  "Tensor " + std::to_string(tensor->getFuid()) +
                      " has null backing memory");
    }
}

void GraphObj::freeHeap() {
    const bool changed = allocator.getHeapPeak() != 0;
    allocator.freeHeap();
    if (changed)
        ++allocationGeneration;
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
