﻿#pragma once

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

    Tensor conv(Tensor input, Tensor weight, Tensor bias, Tensor output, int ph,
                int pw, int sh, int sw, int dh, int dw);
    Tensor convTransposed2d(Tensor input, Tensor weight, Tensor output, int ph,
                            int pw, int sh, int sw, int dh, int dw, int oph,
                            int opw);
    Tensor matmul(Tensor a, Tensor b, Tensor y, bool transA, bool transB,
                  Tensor bias, ActType act);
    Tensor batchNormalization(Tensor input, Tensor output, Tensor mean,
                              Tensor var, Tensor scale, Tensor bias,
                              float momentum, float eps, bool training);

    Tensor maxPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw);
    Tensor avgPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw);

    Tensor unary(std::string ty, Tensor x, Tensor y);
    Tensor binary(std::string ty, Tensor x, Tensor y, Tensor z);
    Tensor softmax(Tensor x, Tensor y, int axis);
    Tensor flatten(Tensor s, Tensor y, int axis);
    Tensor pRelu(Tensor x, Tensor slope, Tensor y);
    Tensor clip(Tensor x, Tensor y, std::optional<float> min,
                std::optional<float> max);
    Tensor transpose(Tensor data, Tensor transposed, Shape perm);
    Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
    Tensor concat(TensorVec inputs, Tensor output, int dim);
    TensorVec split(Tensor input, std::optional<TensorVec> outputs, int axis,
                    int num_outputs);
    Tensor gather(Tensor data, Tensor indices, Tensor output, int axis);
    Tensor reduceMean(Tensor data, Tensor reduced,
                      const optional<vector<int>> &axes, bool keepdims);
    Tensor slice(Tensor input, Tensor output, const vector<int> &starts,
                 const vector<int> &ends, const optional<vector<int>> &axes,
                 const optional<vector<int>> &steps);
    Tensor pad(Tensor input, Tensor output, const vector<int> &pads,
               const optional<vector<int>> &axes);

    //------ modifiers

    inline bool topo_sort() { return g->topo_sort(); }

    inline void optimize() { g->optimize(); }

    //------ runtime

    inline void data_malloc() { g->dataMalloc(); }

    inline void tune() { g->getRuntime()->run(g, true); }

    inline void run() { g->getRuntime()->run(g); }

    inline double get_perf_time() { return g->getRuntime()->getPerfTime(g); }
};

} // namespace infini
