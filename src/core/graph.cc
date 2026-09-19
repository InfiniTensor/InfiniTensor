#include "core/graph.h"
#include "operators/reshape.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <queue>
#include <tuple>

namespace infini {

void GraphCaptureStateObj::applyPendingChanges() noexcept {
    if (!pendingLayoutChange && !pendingStorageChange && !pendingTopologyChange)
        return;
    ++generation;
    if (pendingTopologyChange)
        ++topologyEpoch;
    if (pendingStorageChange || pendingTopologyChange)
        runtime->invalidateGraphCaptureCache(id);
    pendingLayoutChange = false;
    pendingStorageChange = false;
    pendingTopologyChange = false;
}

void GraphCaptureStateObj::beginUpdate() { ++updateDepth; }

void GraphCaptureStateObj::commitUpdate() noexcept {
    if (updateDepth == 0)
        return;
    --updateDepth;
    if (updateDepth == 0)
        applyPendingChanges();
}

void GraphCaptureStateObj::finishMemoryUpdate(bool layoutChanged,
                                              bool storageChanged) {
    IT_ASSERT(updateDepth == 1, "Unexpected nested capture state update");
    pendingLayoutChange = pendingTopologyChange || layoutChanged;
    pendingStorageChange = pendingTopologyChange || storageChanged;
    commitUpdate();
}

void GraphCaptureStateObj::markChanged(bool storageChanged) noexcept {
    pendingLayoutChange = true;
    pendingStorageChange = pendingStorageChange || storageChanged;
    if (updateDepth == 0)
        applyPendingChanges();
}

void GraphCaptureStateObj::markTopologyChanged() noexcept {
    pendingLayoutChange = true;
    pendingStorageChange = true;
    pendingTopologyChange = true;
    if (updateDepth == 0)
        applyPendingChanges();
}

GraphObj::GraphObj(Runtime runtime, OpVec ops_in)
    : runtime(runtime), allocator(runtime),
      captureState(make_ref<GraphCaptureStateObj>(guid, runtime)),
      sorted(false) {
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

GraphObj::~GraphObj() {
    captureState->markTopologyChanged();
    for (const auto &tensor : tensors)
        tensor->unregisterCaptureState(captureState->getId());
}

void GraphObj::registerTensorCaptureState(const Tensor &tensor) {
    tensor->registerCaptureState(captureState);
}

void GraphObj::markTopologyChanged() {
    sorted = false;
    captureState->markTopologyChanged();
}

void GraphObj::addOperatorAndConnect(const Operator &op) {
    markTopologyChanged();
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
    foldFixedShapeSubgraph();
    // After the fold, because what it leaves behind is what may still be
    // duplicated: a chain that settled is gone entirely, and two copies of one
    // that did not are still two operators reading the same tensors.
    mergeDuplicateShapeOperators();
    for (auto &op : ops) {
        switch (op->getOpType().underlying()) {
        default:
            break;
        }
    }
}

namespace {
/// Whether `op` only reports, selects or joins dimensions, so that its result
/// is worked out during shape inference rather than by running anything.
bool describesShapes(const Operator &op) {
    switch (op->getOpType().underlying()) {
    case OpType::Shape:
    case OpType::Gather:
    case OpType::Unsqueeze:
    case OpType::Squeeze:
    case OpType::Slice:
    case OpType::Concat:
    case OpType::Identity:
    // The arithmetic a shape computation is built from. Each of these is a
    // pure function of its inputs, so one whose result nobody reads has no
    // effect left to lose.
    case OpType::Add:
    case OpType::Sub:
    case OpType::Mul:
    case OpType::Div:
    case OpType::Max:
    case OpType::Min:
    case OpType::FloorDiv:
    case OpType::FloorMod:
        return true;
    default:
        return false;
    }
}

/// Write a settled shape value into the tensor itself.
///
/// Shape inference leaves the result beside the tensor rather than in it,
/// which is enough for a `Reshape` -- that reads the value while shapes are
/// worked out. It is not enough for an operator that still runs: a `Concat`
/// joining a settled dimension with one that still moves reads its inputs
/// through a kernel, and would find whatever the memory last held. Nothing
/// runs to fill this tensor once its producer is gone, so it is filled here.
void writeShapeValueAsData(const Tensor &tensor) {
    const auto &value = *tensor->getShapeValue();
    // Nothing runs to fill this tensor once its producer is gone, so it has to
    // keep what is written here for good. Storage from the pool does not last
    // that long: a tensor with no producer is given an offset and a count of
    // its readers, and the offset is handed out again once the last of them has
    // run. So a plain intermediate is given storage of its own and marked a
    // weight, which is what allocation calls memory it must leave alone.
    //
    // A tensor the graph takes in or gives back already has storage that is
    // never reused, and marking one a weight would take away the very thing
    // that makes it an input or an output, so those keep what they have.
    const bool needsStorageOfItsOwn = tensor->isOthers();
    if (needsStorageOfItsOwn) {
        tensor->freeData();
    }
    if (!tensor->hasData()) {
        tensor->dataMalloc();
    }
    if (tensor->getDType() == DataType::Int64) {
        tensor->copyin(vector<int64_t>(value.begin(), value.end()));
    } else {
        // `canHoldShapeValue` allows only the two integer types, so this is
        // the other one.
        IT_ASSERT(tensor->getDType() == DataType::Int32);
        tensor->copyin(vector<int32_t>(value.begin(), value.end()));
    }
    if (needsStorageOfItsOwn) {
        tensor->setWeight();
    }
}
} // namespace

size_t GraphObj::shapeSubgraphSize() const {
    // The same test the fold applies, so that the two cannot come to disagree
    // about what a shape operator is. Both halves are needed: the type says
    // which operators a shape computation is built from, and carrying a shape
    // value says this one is part of such a computation rather than of the
    // model. The types are shared -- an attention export squeezes and scales
    // activations with the very operators an exporter joins dimensions with --
    // so the type alone counts arithmetic on data as though shapes could be
    // folded out of it.
    return static_cast<size_t>(
        std::count_if(ops.begin(), ops.end(), [](const Operator &op) {
            if (!describesShapes(op)) {
                return false;
            }
            const auto &outputs = op->getOutputs();
            return !outputs.empty() &&
                   std::all_of(outputs.begin(), outputs.end(),
                               [](const Tensor &t) {
                                   return t->getShapeValue().has_value();
                               });
        }));
}

size_t GraphObj::foldFixedShapeSubgraph() {
    foldedAwayTensors.clear();

    OpVec foldable;
    for (const auto &op : ops) {
        if (!describesShapes(op)) {
            continue;
        }
        // Every output must already hold its final contents. A `Concat`
        // joining a settled dimension with one that still moves is not
        // foldable, and reports itself as such by leaving that element
        // unfixed.
        const auto &outputs = op->getOutputs();
        if (outputs.empty()) {
            continue;
        }
        if (!std::all_of(outputs.begin(), outputs.end(), [](const Tensor &t) {
                return t->isShapeValueWhollyFixed();
            })) {
            continue;
        }
        // Something the graph was asked to produce has to keep whatever
        // produces it. The graph promises that every tensor it holds is either
        // produced by an operator or given from outside, and an output left
        // with neither would break that promise even though the numbers in it
        // are right. What feeds such an operator still folds, so a settled
        // chain collapses to the one operator at its end.
        if (std::any_of(outputs.begin(), outputs.end(),
                        [](const Tensor &t) { return t->isOutput(); })) {
            continue;
        }
        foldable.emplace_back(op);
    }

    // A tensor the graph can no longer reach: nothing produces it and nothing
    // reads it. Such a tensor breaks the graph invariant, so it is let go of --
    // unless it is one of the graph's own inputs or outputs, which stay
    // whatever happens around them. Any reference held elsewhere keeps the
    // value readable; it is only the graph that lets go.
    const auto releaseIfUnreachable = [this](const Tensor &tensor) {
        if (!tensor->getTargets().empty() || tensor->getSource()) {
            return;
        }
        if (tensor->isInput() || tensor->isOutput()) {
            return;
        }
        foldedAwayTensors.push_back(tensor->getFuid());
        removeTensor(tensor);
    };

    // Taking an operator out of the graph, having settled what happens to what
    // it produced. Its inputs lose a reader, which may leave whatever computed
    // them working for nobody, so those producers come back as candidates.
    OpVec pending;
    const auto detach = [&](const Operator &op) {
        const auto inputs = op->getInputs();
        const auto outputs = op->getOutputs();
        for (const auto &output : outputs) {
            output->setSource(Operator{});
            // The operator is about to go, so no consumer may still name it as
            // what comes before them.
            for (const auto &consumer : output->getTargets()) {
                consumer->removePredecessors(op);
            }
        }
        for (const auto &input : inputs) {
            deleteConnection(input, op);
        }
        removeOperator(op);
        for (const auto &output : outputs) {
            releaseIfUnreachable(output);
        }
        for (const auto &input : inputs) {
            if (input->getTargets().empty()) {
                if (const auto producer = input->getSource()) {
                    pending.emplace_back(producer);
                }
            }
            releaseIfUnreachable(input);
        }
    };

    size_t dropped = 0;
    for (const auto &op : foldable) {
        // The output keeps its place in the graph and merely loses its
        // producer. Nothing downstream is rewired, so a consumer cannot be
        // missed, and a reference held elsewhere stays valid. A result nobody
        // reads is not worth a constant; asked for as an output of the graph it
        // is read by whoever asked, so it gets one.
        for (const auto &output : op->getOutputs()) {
            if (!output->getTargets().empty() || output->isInput() ||
                output->isOutput()) {
                writeShapeValueAsData(output);
            }
        }
        detach(op);
        ++dropped;
    }

    // Whatever the folds above left with nothing to feed. An operator only
    // goes if everything it produces is unread -- one consumer left anywhere
    // means it is still doing work -- and only if it is one of the operators
    // this pass understands, so that nothing with an effect of its own is
    // dropped for looking unused.
    while (!pending.empty()) {
        const auto op = pending.back();
        pending.pop_back();
        if (std::find(ops.begin(), ops.end(), op) == ops.end()) {
            continue;
        }
        if (!describesShapes(op)) {
            continue;
        }
        const auto &outputs = op->getOutputs();
        if (!std::all_of(outputs.begin(), outputs.end(), [](const Tensor &t) {
                return t->getTargets().empty() && !t->isInput() &&
                       !t->isOutput();
            })) {
            continue;
        }
        detach(op);
        ++dropped;
    }

    // Mixed tensors stay intact. A fixed Gather or Slice result already folds
    // above; splitting its input while retaining the producer only adds work.
    return dropped;
}

size_t GraphObj::mergeDuplicateShapeOperators() {
    IT_ASSERT(topo_sort() == true);

    // What makes two of these operators the same computation: the same kind of
    // operator, reading the very same tensors, with the same attributes. The
    // tensors are compared as objects rather than by what they currently hold,
    // so two that happen to carry equal numbers under this shape are not taken
    // for one another -- they may part company under the next.
    //
    // The attributes come from the vector each operator already reports for
    // scheduling, which is exactly the list of what distinguishes it from
    // another of its kind. Reading them from there rather than naming them
    // here means an operator that gains an attribute cannot quietly start
    // being merged with one that differs in it.
    using Signature =
        std::tuple<OpType::underlying_t, vector<UidBaseType>, vector<int>>;
    const auto signatureOf = [](const Operator &op) {
        vector<UidBaseType> inputs;
        inputs.reserve(op->getInputs().size());
        for (const auto &input : op->getInputs())
            inputs.push_back(input->getFuid());
        return Signature{op->getOpType().underlying(), std::move(inputs),
                         op->getOpAttrVector()};
    };

    // Whether this operator may stand in for an earlier one, or be replaced by
    // it. Only shape computations are considered, for the reason the fold
    // gives: these types serve data as readily as dimensions, and carrying a
    // shape value is what says this one describes dimensions.
    const auto isMergeable = [](const Operator &op) {
        if (!describesShapes(op)) {
            return false;
        }
        const auto &outputs = op->getOutputs();
        if (outputs.empty()) {
            return false;
        }
        return std::all_of(outputs.begin(), outputs.end(), [](const Tensor &t) {
            // Something the graph was asked for keeps its own producer:
            // dropping it would leave the graph owing a tensor nothing
            // produces. One it was given is not produced here at all.
            return t->getShapeValue().has_value() && !t->isInput() &&
                   !t->isOutput();
        });
    };

    size_t merged = 0;
    // Merging a pair makes their consumers read one tensor where they read two,
    // which is what makes those consumers duplicates in turn: the second
    // `Gather` of a pair only becomes a copy of the first once both read the
    // same `Shape` output. So the scan repeats until a pass finds nothing, and
    // a chain of any length collapses rather than only its first link.
    for (;;) {
        std::map<Signature, Operator> canonical;
        vector<std::pair<Operator, Operator>> merges;
        for (const auto &op : ops) {
            if (!isMergeable(op)) {
                continue;
            }
            const auto [it, inserted] =
                canonical.try_emplace(signatureOf(op), op);
            if (inserted) {
                continue;
            }
            const auto &keep = it->second;
            if (keep->getOutputs().size() != op->getOutputs().size()) {
                continue;
            }
            merges.emplace_back(op, keep);
        }
        if (merges.empty()) {
            break;
        }

        for (const auto &[drop, keep] : merges) {
            const auto &dropOutputs = drop->getOutputs();
            const auto &keepOutputs = keep->getOutputs();
            for (size_t i = 0; i < dropOutputs.size(); ++i) {
                // Whoever read the copy now reads the one that stays. The
                // consumers are copied first: rewiring one takes it off this
                // very list.
                const auto consumers = dropOutputs[i]->getTargets();
                for (const auto &consumer : consumers)
                    replaceConnection(dropOutputs[i], keepOutputs[i], consumer);
            }
            for (const auto &output : dropOutputs) {
                output->setSource(Operator{});
                IT_ASSERT(output->getTargets().empty(),
                          "a merged output still has a reader");
                foldedAwayTensors.push_back(output->getFuid());
                removeTensor(output);
            }
            for (const auto &input : drop->getInputs())
                deleteConnection(input, drop);
            removeOperator(drop);
            ++merged;
        }
    }
    return merged;
}

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::spreadFixedDims(const Operator &op) {
    const auto &inputs = op->getInputs();
    const auto &outputs = op->getOutputs();
    for (size_t o = 0; o < outputs.size(); ++o) {
        const auto rank = outputs[o]->getRank();
        DimDescs descs(rank, DimDesc{false, ""});
        for (size_t d = 0; d < rank; ++d) {
            // A dimension follows some input dimensions, and can change exactly
            // when one of those can. Naming none of them says it follows
            // nothing a caller may vary.
            for (const auto &source : op->dimSources(o, d)) {
                IT_ASSERT(source.input < inputs.size());
                if (inputs[source.input]->isDimDynamic(source.dim)) {
                    descs[d].dynamic = true;
                    break;
                }
            }
        }
        // A tensor with every dimension dynamic is the same thing as one that
        // declared nothing, and that is how such a tensor has always been left.
        const bool anyFixed =
            std::any_of(descs.begin(), descs.end(),
                        [](const DimDesc &d) { return !d.dynamic; });
        outputs[o]->setDimDescs(anyFixed ? std::move(descs) : DimDescs{});
    }
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
        // Which of these dimensions can change follows from which of the ones
        // they were worked out from can. This goes after the shapes are in
        // place, because a `Reshape` may have just given an output a different
        // rank and a description has to have one per dimension.
        spreadFixedDims(op);
        // Shapes are settled for this operator, so a shape value that follows
        // from them can be worked out now. `ops` is in topological order, so
        // every consumer is visited after its producers and sees a current
        // value rather than one left over from earlier shapes.
        op->inferShapeValue();
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
    struct CaptureMemoryState {
        uint64_t storageId;
        size_t storageOffset;
        size_t blobBytes;
        const void *address;

        bool operator==(const CaptureMemoryState &other) const {
            return storageId == other.storageId &&
                   storageOffset == other.storageOffset &&
                   blobBytes == other.blobBytes && address == other.address;
        }
        bool operator!=(const CaptureMemoryState &other) const {
            return !(*this == other);
        }
    };
    const auto getCaptureMemoryState = [](const Tensor &tensor) {
        const auto &blob = tensor->getDataBlob();
        if (!blob)
            return CaptureMemoryState{0, 0, 0, nullptr};
        return CaptureMemoryState{blob->getStorageId(),
                                  blob->getStorageOffset(), blob->getBytes(),
                                  tensor->getRawDataPtr<const void *>()};
    };
    vector<CaptureMemoryState> previousCaptureStates;
    previousCaptureStates.reserve(tensors.size());
    for (const auto &tensor : tensors)
        previousCaptureStates.emplace_back(getCaptureMemoryState(tensor));
    const auto finishCaptureStateUpdate = [&]() {
        bool layoutChanged = previousCaptureStates.size() != tensors.size();
        bool storageChanged = layoutChanged;
        for (size_t i = 0;
             i < previousCaptureStates.size() && i < tensors.size(); ++i) {
            const auto current = getCaptureMemoryState(tensors[i]);
            layoutChanged =
                layoutChanged || previousCaptureStates[i] != current;
            storageChanged =
                storageChanged ||
                previousCaptureStates[i].storageId != current.storageId;
        }
        captureState->finishMemoryUpdate(layoutChanged, storageChanged);
    };

    captureState->beginUpdate();
    if (allocationMode != AllocationMode::Uninitialized) {
        try {
            dataMallocImplCore(useNaiveAllocator, memPoolSize, trim);
            finishCaptureStateUpdate();
        } catch (...) {
            finishCaptureStateUpdate();
            throw;
        }
        return;
    }

    vector<Blob> previousData;
    previousData.reserve(tensors.size());
    for (const auto &tensor : tensors)
        previousData.emplace_back(tensor->getDataBlob());

    const auto previousGeneration = allocationGeneration;
    try {
        dataMallocImplCore(useNaiveAllocator, memPoolSize, trim);
        finishCaptureStateUpdate();
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
        finishCaptureStateUpdate();
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
        if (!weightAllocated) {
            weightAllocated = true;
            ++weightDataGeneration;
        }
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
        ++weightDataGeneration;
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
            captureState->markChanged(true);
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
    if (changed)
        captureState->markChanged(true);
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
    auto tensor = make_ref<TensorObj>(dim, dtype, runtime);
    registerTensorCaptureState(tensor);
    tensors.emplace_back(tensor);
    markTopologyChanged();
    return tensor;
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                  tensor->getRuntime()->toString() + " to " +
                  runtime->toString());
    tensors.emplace_back(tensor);
    registerTensorCaptureState(tensor);
    markTopologyChanged();
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

void GraphObj::removeOperator(Operator op) {
    auto it = std::find(ops.begin(), ops.end(), op);
    if (it == ops.end())
        return;
    ops.erase(it);
    markTopologyChanged();
}

void GraphObj::removeTensor(Tensor tensor) {
    auto it = std::find(tensors.begin(), tensors.end(), tensor);
    if (it == tensors.end())
        return;
    tensor->unregisterCaptureState(captureState->getId());
    tensors.erase(it);
    markTopologyChanged();
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
    markTopologyChanged();
}

// add op as a target
void GraphObj::addConnection(Tensor tensor, Operator op) {
    tensor->addTarget(op);
    if (tensor->getSource()) {
        tensor->getSource()->addSuccessors(op);
        op->addPredecessors(tensor->getSource());
    }
    markTopologyChanged();
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
