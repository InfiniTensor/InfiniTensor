#pragma once

#include "core/graph.h"
#include "core/runtime.h"
#include <cstdint>
#include <iostream>

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
                  Tensor bias, ActType act);
    Tensor batchNormalization(Tensor input, Tensor output, Tensor mean,
                              Tensor var, Tensor scale, Tensor bias,
                              float momentum, float eps, bool training);
    Tensor layerNormalization(Tensor input, Tensor scale, Tensor output,
                              Tensor bias, float eps, int axis, int stash_type);

    Tensor maxPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw, int ceilMode);
    Tensor avgPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw, int ceilMode);

    Tensor add(Tensor a, Tensor b, Tensor c);
    Tensor sub(Tensor a, Tensor b, Tensor c);
    Tensor mul(Tensor a, Tensor b, Tensor c);
    Tensor div(Tensor a, Tensor b, Tensor c);
    Tensor pow(Tensor a, Tensor b, Tensor c);
    Tensor min(Tensor a, Tensor b, Tensor c);
    Tensor max(Tensor a, Tensor b, Tensor c);

    Tensor relu(Tensor x, Tensor y);
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
    Tensor clip(Tensor x, Tensor y, std::optional<float> min,
                std::optional<float> max);
    Tensor transpose(Tensor data, Tensor transposed, Shape perm);
    Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
    Tensor concat(TensorVec inputs, Tensor output, int dim);
    Tensor attentionKVCache(Tensor input_k_cache, Tensor input_v_cache,
                            Tensor input_q, Tensor input_k, Tensor input_v,
                            Tensor position_id, Tensor output_matmul);
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
    Tensor pad(Tensor input, Tensor output, const vector<int> &pads,
               const optional<vector<int>> &axes);
    Tensor cast(Tensor input, Tensor output, int to);
    Tensor expand(Tensor input, Tensor output, Shape dims);
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

    inline void shape_infer() { g->shape_infer(); }

    void change_shape(const vector<int> &shape, int tensorId);
    //------ runtime

    inline void data_malloc(bool useNaiveAllocator = false,
                            size_t memPoolSize = 0) {
        g->dataMalloc(useNaiveAllocator, memPoolSize);
    }

    inline Tensor clone_KV(Tensor &tensor) { return g->cloneKV(tensor); }

    inline void free_heap() { g->freeHeap(); }

    inline void tune() { g->getRuntime()->run(g, true); }

    inline void run() { g->getRuntime()->run(g); }

    inline double get_perf_time() { return g->getRuntime()->getPerfTime(g); }
};

} // namespace infini
