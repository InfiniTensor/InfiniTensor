#pragma once

#include "core/graph.h"
#include "core/operator.h"
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
                int sh, int sw, int dh, int dw, Tensor bias = nullptr,
                bool channelsLast = false);
    bool supportsConvBias() const;
    bool supportsChannelsLastConv() const;
    Tensor conv3d(Tensor input, Tensor weight, Tensor output, int pd, int ph,
                  int pw, int sd, int sh, int sw, int dd, int dh, int dw);
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
    Tensor rmsNorm(Tensor input, Tensor weight, Tensor output, float epsilon = 1e-6f);

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
    Tensor leakyRelu(Tensor x, Tensor y, float alpha);
    Tensor silu(Tensor x, Tensor y);
    Tensor gelu(Tensor x, Tensor y);
    Tensor sigmoid(Tensor x, Tensor y);
    Tensor hardSigmoid(Tensor x, Tensor y, double alpha = 0.2,
                       double beta = 0.5);
    Tensor hardSwish(Tensor x, Tensor y);
    Tensor tanh(Tensor x, Tensor y);
    Tensor erf(Tensor x, Tensor y);
    Tensor sin(Tensor x, Tensor y);
    Tensor cos(Tensor x, Tensor y);
    Tensor softmax(Tensor x, Tensor y, int axis);
    Tensor abs(Tensor x, Tensor y);
    Tensor sign(Tensor x, Tensor y);
    Tensor rsqrt(Tensor x, Tensor y);
    Tensor log(Tensor x, Tensor y);
    Tensor sqrt(Tensor x, Tensor y);
    Tensor neg(Tensor x, Tensor y);
    Tensor logicalNot(Tensor x, Tensor y);
    Tensor shape(Tensor x, Tensor y);
    Tensor identity(Tensor x, Tensor y);
    Tensor flatten(Tensor s, Tensor y, int axis);
    Tensor pRelu(Tensor x, Tensor slope, Tensor y);
    Tensor elu(Tensor x, Tensor y, float alpha);
    Tensor clip(Tensor x, Tensor y, std::optional<float> min,
                std::optional<float> max);
    Tensor transpose(Tensor data, Tensor transposed, Shape perm);
    Tensor tile(Tensor data, Tensor output, Shape repeats);
    Tensor equal(Tensor lhs, Tensor rhs, Tensor output);
    Tensor less(Tensor lhs, Tensor rhs, Tensor output);
    Tensor greaterEqual(Tensor lhs, Tensor rhs, Tensor output);
    Tensor gridSample(Tensor data, Tensor grid, Tensor output, bool nearest,
                      bool reflection, bool alignCorners);
    Tensor roiAlign(Tensor data, Tensor boxes, Tensor batchIndices, Tensor output,
                    int height, int width, int samplingRatio, float spatialScale, bool aligned);
    Tensor scatterNd(Tensor data, Tensor indices, Tensor updates, Tensor output,
                     bool add);
    Tensor maskedScatter(Tensor data, Tensor mask, Tensor source, Tensor output);
    Tensor scatterElements(Tensor data, Tensor indices, Tensor updates, Tensor output,
                           int axis, std::string reduction);
    Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
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
    Tensor argmax(Tensor data, Tensor output, int axis, bool keepdims);
    Tensor reduceMax(Tensor data, Tensor output,
                     const std::optional<vector<int>> &axes, bool keepDims);
    Tensor reduceL2(Tensor data, Tensor output,
                    const optional<vector<int>> &axes, bool keepdims);
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

    inline void trim_memory() { g->trimMemory(); }

    inline Tensor clone_KV(Tensor &tensor) { return g->cloneKV(tensor); }

    inline void free_heap() { g->freeHeap(); }

    inline void tune() { g->getRuntime()->run(g, true); }

    inline void run() { g->getRuntime()->run(g); }

    inline double get_perf_time() { return g->getRuntime()->getPerfTime(g); }

    inline void run_with_graph() { g->getRuntime()->runWithGraph(g); }
};

} // namespace infini
