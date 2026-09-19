#pragma once
#include "core/lazy_allocator.h"
#include "core/operator.h"
#include "core/tensor.h"
#include <algorithm>
#include <cstdint>

namespace infini {

class GraphCaptureStateObj {
  private:
    uint64_t id;
    Runtime runtime;
    size_t generation = 0;
    size_t topologyEpoch = 0;
    size_t updateDepth = 0;
    bool pendingLayoutChange = false;
    bool pendingStorageChange = false;
    bool pendingTopologyChange = false;

    void applyPendingChanges() noexcept;

  public:
    GraphCaptureStateObj(uint64_t id, Runtime runtime)
        : id(id), runtime(std::move(runtime)) {}

    uint64_t getId() const { return id; }
    size_t getGeneration() const { return generation; }
    size_t getTopologyEpoch() const { return topologyEpoch; }

    void beginUpdate();
    void commitUpdate() noexcept;
    void finishMemoryUpdate(bool layoutChanged, bool storageChanged);
    void markChanged(bool storageChanged) noexcept;
    void markTopologyChanged() noexcept;
};

class GraphObj : public Object {
  protected:
    Runtime runtime;
    TensorVec tensors;
    OpVec ops;
    LazyAllocator allocator;
    Ref<GraphCaptureStateObj> captureState;

  public:
    explicit GraphObj(Runtime runtime)
        : runtime(runtime), allocator(runtime),
          captureState(make_ref<GraphCaptureStateObj>(guid, runtime)),
          sorted(false){};
    GraphObj(Runtime runtime, OpVec ops_in);
    ~GraphObj() override;
    string toString() const override;
    Runtime getRuntime() const { return runtime; }

    Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);
    Tensor addTensor(const Tensor &tensor);
    TensorVec addTensor(const TensorVec &tensors);
    /**
     * @brief Clone a tensor and add it to the graph.
     */
    Tensor cloneTensor(const Tensor &tensor) {
        return addTensor(tensor->clone(runtime));
    }
    void removeOperator(Operator op);

    void removeTensor(Tensor tensor);

    void deleteConnection(Tensor tensor, Operator op);
    void addConnection(Tensor tensor, Operator op);
    void replaceConnection(Tensor oldInput, Tensor newInput, Operator op);

    Operator cloneOperator(Operator op, TensorVec inputs, TensorVec outputs) {
        auto opClone = op->clone(inputs, outputs);
        addOperatorAndConnect(opClone);
        return opClone;
    }

    const TensorVec &getTensors() const { return tensors; }
    const OpVec &getOperators() const { return ops; }
    OpVec getComputeOps() const;
    Tensor getTensor(int) const;

    /**
     * Sort the nodes in topological order.
     * It returns true if the sorting is successful.
     * Otherwise false is returned, means that there are rings in the graph,
     * so the topological sorting fails.
     */
    bool topo_sort();

    void optimize();

    /// @brief Replaces the settled part of the shape subgraph with constants.
    ///
    /// An operator whose result is the same under every shape the graph may
    /// legally be given need not be run, nor re-run whenever the shape
    /// changes. Its output already holds that result, worked out during shape
    /// inference, so the operator is dropped and its output left standing as a
    /// constant. See `TensorObj::isShapeValueFixed`.
    ///
    /// @return How many operators were dropped.
    size_t foldFixedShapeSubgraph();

    /// @brief Have operators that describe the same shape once rather than
    /// repeatedly.
    ///
    /// An exporter writing one dimension into several places emits the same
    /// small computation once for each of them: three `Unsqueeze` operators on
    /// the very same tensor with the very same axes, whose outputs cannot but
    /// agree. One of each such set is kept and its consumers moved onto it.
    ///
    /// Two operators are the same when their type, their attributes and the
    /// tensor objects they read are all the same -- not when the numbers they
    /// currently hold happen to match, which says only what this shape made of
    /// them. That leaves this sound for a computation that still moves: reading
    /// the same input through the same operator gives the same answer under
    /// every shape, so what is dropped is the repetition and never the work.
    ///
    /// Distinct from `foldFixedShapeSubgraph`, and useful after it: a fold
    /// removes what has settled, while this removes what is duplicated, and a
    /// duplicated computation that still follows a dynamic dimension is beyond
    /// a fold's reach. Runs only over operators that describe shapes, so
    /// nothing with an effect of its own is touched.
    ///
    /// @return How many operators were dropped.
    size_t mergeDuplicateShapeOperators();

    /// @brief How many operators of this graph describe shapes rather than
    /// data, which is the part of it `foldFixedShapeSubgraph` can reach. Says
    /// what a fold started from, so that what it left can be compared against
    /// it.
    ///
    /// An operator counts when its type is one a shape computation is built
    /// from and it carries a shape value. The type on its own is not enough,
    /// because those types serve data as readily as dimensions: an operator
    /// scaling activations is not something a fold could ever reach, and
    /// counting it would report a shape subgraph larger than the one that
    /// exists.
    size_t shapeSubgraphSize() const;

    /// @brief Tensors dropped by `foldFixedShapeSubgraph` as no longer
    /// reachable, by fuid. A caller holding its own references -- the ONNX
    /// importer keeps one per initializer -- needs to let go of these.
    const vector<UidBaseType> &getFoldedAwayTensors() const {
        return foldedAwayTensors;
    }

    void shape_infer();

    /// Say which dimensions of what `op` computes are fixed.
    ///
    /// `dimDescs` says which dimensions of a tensor a graph is given may
    /// change; nothing said it of one the graph works out. A computed shape is
    /// a function of the shapes the graph was given -- `inferShape` and nothing
    /// else -- so a dimension of a computed tensor can change exactly when some
    /// dimension it was worked out from can. Which makes this a matter of
    /// following the operators rather than of asking each one what it does.
    ///
    /// Call this once `op` has its output shapes, and in topological order, so
    /// that what it reads of its inputs is current.
    void spreadFixedDims(const Operator &op);

    void dataMalloc(bool useNaiveAllocator = false, size_t memPoolSize = 0);

    void trimMemory();

    void validateMemory() const;

    size_t getAllocationGeneration() const { return allocationGeneration; }

    /// Which time over this graph's life the weights were given their storage.
    /// Weight storage is handed out once and then left alone, so whoever wrote
    /// the weights can tell that what they wrote still stands by finding this
    /// unchanged, and need not write it again. It moves only when the weights
    /// are given storage afresh, which the generation above cannot say: that
    /// one counts activation layouts and so moves on almost every shape.
    size_t getWeightDataGeneration() const { return weightDataGeneration; }

    /// How many times activation storage has been asked of the runtime over
    /// this graph's life. Distinct from the generation above, which counts
    /// layouts: shrinking a tensor gives it a new size at a new offset without
    /// any memory being asked for, so the generation moves and this does not.
    /// It is this one that says whether a series of shapes is paying for its
    /// changes.
    size_t getActivationAllocations() const {
        return allocator.getActivationAllocations();
    }

    /// What the activations need, against what is held for them, in bytes.
    /// Capacity stands at the high watermark of the shapes seen so far, and the
    /// difference is the slack that lets a smaller shape reuse the storage.
    size_t getActivationPeak() const { return allocator.getActivationPeak(); }
    size_t getActivationCapacity() const {
        return allocator.getActivationCapacity();
    }

    /// Every byte the graph holds: the activation capacity as actually
    /// allocated, the weights, and the heap that cloned tensors live on.
    size_t getAllocatedBytes() const {
        return allocator.getActivationCapacity() + allocator.getWeightPeak() +
               allocator.getHeapPeak();
    }

    uint64_t getCaptureStateId() const { return captureState->getId(); }
    size_t getCaptureGeneration() const {
        return captureState->getGeneration();
    }
    size_t getTopologyEpoch() const { return captureState->getTopologyEpoch(); }

    Tensor cloneKV(Tensor &tensor);

    void freeHeap();

    /**
     * @brief Add an operator and create its outputs. Output tensor arguments
     * should be empty Refs (e.g., nullptr).
     */
    template <typename T, typename... Args> Ref<T> addOp(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    /**
     * @brief Add an operator with its outputs specified.
     */
    template <typename T, typename... Args>
    Ref<T> addOpWithOutputs(Args &&...args) {
        Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
        addOperatorAndConnect(op);
        return op;
    }

    /**
     * @brief Gets input tensors of this graph.
     */
    inline TensorVec getInputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (!t->getSource())
                ret.emplace_back(t);
        return ret;
    }

    /**
     * @brief Gets output tensors of this graph.
     */
    inline TensorVec getOutputs() const {
        TensorVec ret;
        for (const auto &t : tensors)
            if (t->getTargets().empty())
                ret.emplace_back(t);
        return ret;
    }

    bool checkValid() const;

  private:
    enum class AllocationMode {
        Uninitialized,
        Naive,
        DynamicPool,
        FixedPool,
    };

    void dataMallocImpl(bool useNaiveAllocator, size_t memPoolSize, bool trim);

    void dataMallocImplCore(bool useNaiveAllocator, size_t memPoolSize,
                            bool trim);

    void lockAllocationMode(bool useNaiveAllocator, size_t memPoolSize);

    void registerTensorCaptureState(const Tensor &tensor);
    void markTopologyChanged();

    /**
     * @brief Add reverse connections and Op relationship in ctor.
     */
    void addOperatorAndConnect(const Operator &op);

    /**
     * @brief If the nodes is sorted in topological order.
     */
    bool sorted;

    /**
     * @brief If the weight tensors are allocated.
     */
    bool weightAllocated = false;

    AllocationMode allocationMode = AllocationMode::Uninitialized;
    size_t fixedPoolSize = 0;

    /**
     * @brief Incremented after each successful memory layout update.
     */
    size_t allocationGeneration = 0;

    /**
     * @brief Incremented whenever the weights are given their storage, which
     * `weightAllocated` says happens once until something takes that storage
     * away again. Never wound back: a rollback leaves the weights needing to be
     * written afresh, and a count that returned to a value someone recorded
     * earlier would tell them the opposite.
     */
    size_t weightDataGeneration = 0;

    /**
     * Fixed memory pools cannot safely move live data within the same backing
     * allocation. Remember the committed layout so changes can be rejected
     * before any tensor data is modified.
     */
    vector<UidBaseType> foldedAwayTensors;

    bool fixedPoolLayoutCommitted = false;
    vector<std::pair<TensorObj *, size_t>> fixedPoolTensorLayout;
    vector<std::pair<TensorObj *, size_t>> fixedPoolActivationLayout;
};

} // namespace infini
