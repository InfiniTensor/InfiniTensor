#pragma once

#include "core/graph.h"
#include "core/runtime.h"
#include <cstdint>
#include <iostream>

namespace infini {

// Use the indices from onnx to reduce delivery overhead,
// which comes from onnx but may be not only used for onnx.
//
// see https://onnx.ai/onnx/intro/concepts.html#element-type
enum OnnxDType : int {
    UNDEFINED = 0,
    FLOAT,
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    INT64,
    STRING,
    BOOL,
    FLOAT16,
    DOUBLE,
    UINT32,
    UINT64,
    COMPLEX64,
    COMPLEX128,
    BFLOAT16,
};

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
    Tensor matmul(Tensor a, Tensor b, Tensor y, bool transA, bool transB,
                  Tensor bias, ActType act);
    Tensor batchNorm(Tensor input, Tensor output, Tensor mean, Tensor var,
                     Tensor scale, Tensor bias, float momentum, float eps,
                     bool training);

    Tensor maxPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw);
    Tensor avgPool(Tensor input, Tensor output, int kh, int kw, int dh, int dw,
                   int ph, int pw, int sh, int sw);

    Tensor add(Tensor a, Tensor b, Tensor c);
    Tensor sub(Tensor a, Tensor b, Tensor c);
    Tensor mul(Tensor a, Tensor b, Tensor c);
    Tensor div(Tensor a, Tensor b, Tensor c);
    Tensor pow(Tensor a, Tensor b, Tensor c);

    Tensor relu(Tensor x, Tensor y);
    Tensor sigmoid(Tensor x, Tensor y);
    Tensor tanh(Tensor x, Tensor y);
    Tensor softmax(Tensor x, Tensor y, int axis);
    Tensor abs(Tensor x, Tensor y);
    Tensor clip(Tensor x, Tensor y, float min, float max);
    Tensor identity(Tensor x, Tensor y);
    Tensor flatten(Tensor s, Tensor y, int axis);
    Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
    Tensor concat(TensorVec inputs, Tensor output, int dim);
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

    //------ runtime

    inline void data_malloc() { g->dataMalloc(); }

    inline void run() { g->getRuntime()->run(g); }
};

} // namespace infini
