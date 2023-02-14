#pragma once

#include "core/graph.h"
#include "core/runtime.h"

namespace infini {

// 借用 onnx 的定义减小传递开销，来自 onnx 但不只用于 onnx。
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

    Tensor matmul(Tensor a, Tensor b, Tensor y, bool transA, bool transB,
                  Tensor bias, ActType act);

    Tensor batchNorm(Tensor input, Tensor output, Tensor mean, Tensor var,
                     Tensor scale, Tensor bias, float momentum, float eps,
                     bool training);

    Tensor add(Tensor a, Tensor b, Tensor c);
    Tensor sub(Tensor a, Tensor b, Tensor c);
    Tensor mul(Tensor a, Tensor b, Tensor c);
    Tensor div(Tensor a, Tensor b, Tensor c);
    Tensor pow(Tensor a, Tensor b, Tensor c);

    Tensor relu(Tensor x, Tensor y);
    Tensor sigmoid(Tensor x, Tensor y);
    Tensor tanh(Tensor x, Tensor y);
    Tensor softmax(Tensor x, Tensor y);
    Tensor abs(Tensor x, Tensor y);
    Tensor identity(Tensor x, Tensor y);
    Tensor flatten(Tensor s, Tensor y);
    Tensor reshape(Tensor data, Tensor reshaped, Shape shape);
};

} // namespace infini
