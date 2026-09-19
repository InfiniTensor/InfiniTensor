#pragma once

#include "core/graph.h"
#include "core/operator.h"
#include "core/runtime.h"
#include <cstdint>
#include <iostream>

#ifdef USE_CUDA
#include "cuda/cuda_runtime.h"
#endif

namespace infini {

class GraphHandlerObj {
    Graph g;

  public:
    GraphHandlerObj(Runtime runtime)
        : g(make_ref<GraphObj>(std::move(runtime))) {}

    Tensor tensor(Shape dims, int dtype);

    //------ operators

    inline OpVec operators() { return g->getOperators(); }

    Tensor conv(Tensor input, Tensor weight, Tensor output, int ph, int pw,
                int sh, int sw, int dh, int dw);
    Tensor convTransposed2d(Tensor input, Tensor weight, Tensor output, int ph,
                            int pw, int sh, int sw, int dh, int dw, int oph,
                            int opw);
    Tensor matmul(Tensor a, Tensor b, Tensor y, bool transA, bool transB,
                  Tensor bias, ActType act,
                  std::string matmul_compute_type = "default");
    Tensor batchNormalization(Tensor input, Tensor output, Tensor mean,
                              Tensor var, Tensor scale, Tensor bias,
                              float momentum, float eps, bool training);
    Tensor layerNormalization(Tensor input, Tensor scale, Tensor output,
                              Tensor bias, float eps, int axis, int stash_type);
    Tensor instanceNormalization(Tensor input, Tensor output, Tensor scale,
                                 Tensor bias, float eps);
    Tensor rmsNorm(Tensor input, Tensor weight, Tensor output);

    Tensor maxPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw, int ceilMode,
                   bool globalWindow = false);
    /// `globalWindow` says the window is the whole input rather than `kh` by
    /// `kw`, which is what a global pool needs: its spatial size may be
    /// dynamic, and then there is no window to pass until an input arrives.
    Tensor avgPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw, int ceilMode,
                   bool globalWindow = false);

    Tensor add(Tensor a, Tensor b, Tensor c);
    Tensor sub(Tensor a, Tensor b, Tensor c);
    Tensor mul(Tensor a, Tensor b, Tensor c);
    Tensor div(Tensor a, Tensor b, Tensor c);
    Tensor pow(Tensor a, Tensor b, Tensor c);
    Tensor min(Tensor a, Tensor b, Tensor c);
    Tensor max(Tensor a, Tensor b, Tensor c);

    Tensor relu(Tensor x, Tensor y);
    Tensor leakyRelu(Tensor x, Tensor y, float alpha);
    Tensor silu(Tensor x, Tensor y);
    Tensor gelu(Tensor x, Tensor y);
    Tensor sigmoid(Tensor x, Tensor y);
    Tensor hardSigmoid(Tensor x, Tensor y);
    Tensor hardSwish(Tensor x, Tensor y);
    Tensor tanh(Tensor x, Tensor y);
    Tensor erf(Tensor x, Tensor y);
    Tensor softmax(Tensor x, Tensor y, int axis);
    Tensor abs(Tensor x, Tensor y);
    Tensor sqrt(Tensor x, Tensor y);
    Tensor neg(Tensor x, Tensor y);
    Tensor shape(Tensor x, Tensor y);
    Tensor identity(Tensor x, Tensor y);
    Tensor flatten(Tensor s, Tensor y, int axis);
    Tensor pRelu(Tensor x, Tensor slope, Tensor y);
    Tensor elu(Tensor x, Tensor y, float alpha);
    Tensor clip(Tensor x, Tensor y, std::optional<float> min,
                std::optional<float> max);
    Tensor transpose(Tensor data, Tensor transposed, Shape perm);
    Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
    /// @brief Reshapes `data` to the shape another tensor describes, rather
    /// than to a shape known when the graph is built.
    Tensor reshape_with_shape_input(Tensor data, Tensor shape, Tensor reshaped);
    Tensor resize(Tensor input, Tensor output,
                  const std::optional<vector<int>> &axes, Tensor sizes,
                  Tensor scales, Tensor roi, vector<int64_t> sizes_,
                  vector<float> scales_, vector<float> roi_, string mode,
                  string ratioPolicy, string nearestMode,
                  string coordTransMode);
    Tensor squeeze(Tensor input, Tensor output, Shape axes);
    Tensor unsqueeze(Tensor input, Tensor output, Shape axes);
    Tensor concat(TensorVec inputs, Tensor output, int dim);
    Tensor attentionKVCache(Tensor input_k_cache, Tensor input_v_cache,
                            Tensor input_q, Tensor input_k, Tensor input_v,
                            Tensor position_id, Tensor output_matmul);
    Tensor RoPE(Tensor pos, Tensor input, Tensor output);
    TensorVec split(Tensor input, std::optional<TensorVec> outputs, int axis,
                    std::variant<int, vector<int>> numOrRatio);
    Tensor gather(Tensor data, Tensor indices, Tensor output, int axis);
    Tensor gatherElements(Tensor data, Tensor indices, Tensor output, int axis);
    Tensor reduceMean(Tensor data, Tensor reduced,
                      const optional<vector<int>> &axes, bool keepdims);
    Tensor reduceSum(Tensor data, Tensor reduced,
                     const optional<vector<int>> &axes, bool keepdims);
    Tensor slice(Tensor input, Tensor output, const vector<int> &starts,
                 const vector<int> &ends, const optional<vector<int>> &axes,
                 const optional<vector<int>> &steps);
    /// Slices at bounds the graph works out rather than ones the model spells
    /// out. See the matching `SliceObj` constructor.
    Tensor slice_with_bound_inputs(Tensor input, Tensor starts, Tensor ends,
                                   Tensor output,
                                   const optional<vector<int>> &axes,
                                   const optional<vector<int>> &steps);
    Tensor pad(Tensor input, Tensor output, const vector<int> &pads,
               const optional<vector<int>> &axes);
    Tensor cast(Tensor input, Tensor output, int to);
    Tensor expand(Tensor input, Tensor output, Shape dims);
    /// Expands to a shape the graph works out rather than one the model spells
    /// out. See the matching `ExpandObj` constructor.
    Tensor expand_with_shape_input(Tensor input, Tensor shape, Tensor output);
    Tensor tile(Tensor input, Tensor output, Shape repeats);
    /// Repeats by counts the graph works out rather than ones the model spells
    /// out. See the matching `TileObj` constructor.
    Tensor tile_with_repeats_input(Tensor input, Tensor repeats, Tensor output);
    Tensor where(Tensor inputX, Tensor inputY, Tensor condition, Tensor output);
    std::vector<int> getDims(Tensor x) { return x->getDims(); }

    Tensor allReduceSum(Tensor input, Tensor output);
    Tensor allReduceProd(Tensor input, Tensor output);
    Tensor allReduceMin(Tensor input, Tensor output);
    Tensor allReduceMax(Tensor input, Tensor output);
    Tensor allReduceAvg(Tensor input, Tensor output);
    TensorVec allGather(Tensor input, std::optional<TensorVec> outputs, int n);
    Tensor broadcast(Tensor input, Tensor output, int root);
    Tensor send(Tensor input, int source, int destination, Tensor output);
    Tensor recv(Tensor output, int source, int destination, Shape dims,
                int outputType, Tensor input);
    Tensor depthToSpace(Tensor input, Tensor output, int blocksize,
                        std::string mode);
    Tensor lrn(Tensor input, Tensor output, float alpha, float beta, float bias,
               int size);

    //------ modifiers

    inline bool topo_sort() { return g->topo_sort(); }

    inline void optimize() { g->optimize(); }

    inline size_t fold_fixed_shape_subgraph() {
        return g->foldFixedShapeSubgraph();
    }

    inline size_t merge_duplicate_shape_operators() {
        return g->mergeDuplicateShapeOperators();
    }

    inline const vector<UidBaseType> &folded_away_tensors() const {
        return g->getFoldedAwayTensors();
    }

    inline size_t operator_count() const { return g->getOperators().size(); }

    inline size_t shape_subgraph_size() const { return g->shapeSubgraphSize(); }

    inline void shape_infer() { g->shape_infer(); }

    void change_shape(const vector<int> &shape, int tensorId);
    void set_dim_descs(const vector<DimDesc> &descs, int tensorId);
    //------ runtime

    inline void data_malloc(bool useNaiveAllocator = false,
                            size_t memPoolSize = 0) {
        g->dataMalloc(useNaiveAllocator, memPoolSize);
    }

    inline void trim_memory() { g->trimMemory(); }

    inline size_t weight_data_generation() const {
        return g->getWeightDataGeneration();
    }

    inline size_t activation_allocations() const {
        return g->getActivationAllocations();
    }

    inline size_t activation_peak() const { return g->getActivationPeak(); }

    inline size_t activation_capacity() const {
        return g->getActivationCapacity();
    }

    inline size_t allocated_bytes() const { return g->getAllocatedBytes(); }

    inline Tensor clone_KV(Tensor &tensor) { return g->cloneKV(tensor); }

    inline void free_heap() { g->freeHeap(); }

    inline void tune() { g->getRuntime()->run(g, true); }

    inline void run() { g->getRuntime()->run(g); }

    inline double get_perf_time() { return g->getRuntime()->getPerfTime(g); }

#ifdef USE_CUDA
    inline void run_with_cudagraph() {
        (as<CudaRuntimeObj>(g->getRuntime()))->runWithCudaGraph(g);
    }
#endif
};

} // namespace infini
