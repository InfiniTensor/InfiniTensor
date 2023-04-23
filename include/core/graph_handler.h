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
    explicit GraphHandlerObj(Runtime runtime)
        : g(make_ref<GraphObj>(std::move(runtime))) {}

    explicit GraphHandlerObj(Graph g) : g(std::move(g)) {}

    //------ tensors

    vector<Tensor> inputs() { return g->getInputs(); }

    vector<Tensor> outputs() { return g->getOutputs(); }

    Tensor tensor(Shape dims, int dtype, TensorType ttype);

    //------ operators

    OpVec operators() { return g->getOperators(); }

    Tensor conv(Tensor input, Tensor weight, Tensor output, int ph, int pw,
                int sh, int sw, int dh, int dw);
    Tensor convTransposed2d(Tensor input, Tensor weight, Tensor output, int ph,
                            int pw, int sh, int sw, int dh, int dw, int oph,
                            int opw);
    Tensor convNHWC(Tensor input, Tensor weight, Tensor output, int ph, int pw,
                    int sh, int sw, int dh, int dw);
    Tensor convTransposed2dNHWC(Tensor input, Tensor weight, Tensor output,
                                int ph, int pw, int sh, int sw, int dh, int dw,
                                int oph, int opw);
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
    Tensor shape(Tensor x, Tensor y);
    Tensor identity(Tensor x, Tensor y);
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
    /// @brief Import memBound operator from a json
    TensorVec memBound(const TensorVec &inputs, const Tensor &outputs,
                       const string &jsonString);

    //------ modifiers

    bool topo_sort() { return g->topo_sort(); }

    void optimize() { g->optimize(); }

    //------ runtime

    void data_malloc() { g->dataMalloc(); }

    void run() { g->getRuntime()->run(g); }

    Graph getGraph() const;
};

} // namespace infini
