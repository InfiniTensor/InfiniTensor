#include "core/graph_handler.h"
#include "nnet/Visitor/Serializer.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/matmul.h"
#include "operators/membound.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/reduce_mean.h"
#include "operators/reshape.h"
#include "operators/slice.h"
#include "operators/softmax.h"
#include "operators/split.h"
#include "operators/transpose.h"
#include "operators/unary.h"

namespace infini {

static DataType dtype_repr_convert(int);

Tensor GraphHandlerObj::tensor(Shape dims, int dtype, TensorType ttype) {
    return g->addTensor(std::move(dims), dtype_repr_convert(dtype), ttype);
}

Tensor GraphHandlerObj::conv(Tensor input, Tensor weight, Tensor output, int ph,
                             int pw, int sh, int sw, int dh, int dw) {
    if (output) {
        g->addOpWithOutputs<ConvObj>(std::move(input), std::move(weight),
                                     output, ph, pw, sh, sw, dh, dw);
        return output;
    } else {
        return g
            ->addOp<ConvObj>(std::move(input), std::move(weight), output, ph,
                             pw, sh, sw, dh, dw)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::convTransposed2d(Tensor input, Tensor weight,
                                         Tensor output, int ph, int pw, int sh,
                                         int sw, int dh, int dw, int oph,
                                         int opw) {
    if (output) {
        g->addOpWithOutputs<ConvTransposed2dObj>(std::move(input),
                                                 std::move(weight), output, ph,
                                                 pw, sh, sw, dh, dw, oph, opw);
        return output;
    } else {
        return g
            ->addOp<ConvTransposed2dObj>(std::move(input), std::move(weight),
                                         output, ph, pw, sh, sw, dh, dw, oph,
                                         opw)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::convNHWC(Tensor input, Tensor weight, Tensor output,
                                 int ph, int pw, int sh, int sw, int dh,
                                 int dw) {
    if (output) {
        g->addOpWithOutputs<ConvNHWCObj>(std::move(input), std::move(weight),
                                         output, ph, pw, sh, sw, dh, dw);
        return output;
    } else {
        return g
            ->addOp<ConvNHWCObj>(std::move(input), std::move(weight), output,
                                 ph, pw, sh, sw, dh, dw)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::convTransposed2dNHWC(Tensor input, Tensor weight,
                                             Tensor output, int ph, int pw,
                                             int sh, int sw, int dh, int dw,
                                             int oph, int opw) {
    if (output) {
        g->addOpWithOutputs<ConvTransposed2dNHWCObj>(
            std::move(input), std::move(weight), output, ph, pw, sh, sw, dh, dw,
            oph, opw);
        return output;
    } else {
        return g->addOp<ConvTransposed2dNHWCObj>(std::move(input),
                                                 std::move(weight), output, ph,
                                                 pw, sh, sw, dh, dw, oph, opw)
        ->getOutput();
    }
}

Tensor GraphHandlerObj::matmul(Tensor a, Tensor b, Tensor y, bool transA,
                               bool transB, Tensor bias, ActType act) {
    if (y) {
        g->addOpWithOutputs<MatmulObj>(std::move(a), std::move(b), y, transA,
                                       transB, std::move(bias), act);
        return y;
    } else {
        return g
            ->addOp<MatmulObj>(std::move(a), std::move(b), y, transA, transB,
                               std::move(bias), act)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::batchNorm(Tensor input, Tensor output, Tensor mean,
                                  Tensor var, Tensor scale, Tensor bias,
                                  float momentum, float eps, bool training) {
    if (output) {
        g->addOpWithOutputs<BatchNormObj>(
            std::move(input), output, std::move(mean), std::move(var),
            std::move(scale), std::move(bias), momentum, eps, training);
        return output;
    } else {
        return g
            ->addOp<BatchNormObj>(std::move(input), output, std::move(mean),
                                  std::move(var), std::move(scale),
                                  std::move(bias), momentum, eps, training)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::maxPool(Tensor input, Tensor output, int kh, int kw,
                                int dh, int dw, int ph, int pw, int sh,
                                int sw) {
    if (output) {
        g->addOpWithOutputs<MaxPoolObj>(std::move(input), output, kh, kw, dh,
                                        dw, ph, pw, sh, sw);
        return output;
    } else {
        return g
            ->addOp<MaxPoolObj>(std::move(input), output, kh, kw, dh, dw, ph,
                                pw, sh, sw)
            ->getOutput();
    }
}
Tensor GraphHandlerObj::avgPool(Tensor input, Tensor output, int kh, int kw,
                                int dh, int dw, int ph, int pw, int sh,
                                int sw) {
    if (output) {
        g->addOpWithOutputs<AvgPoolObj>(std::move(input), output, kh, kw, dh,
                                        dw, ph, pw, sh, sw);
        return output;
    } else {
        return g
            ->addOp<AvgPoolObj>(std::move(input), output, kh, kw, dh, dw, ph,
                                pw, sh, sw)
            ->getOutput();
    }
}

// see operators/element_wise.h
#define DEFINE_ELEMENT_WISE_METHOD(name, obj)                                  \
    Tensor GraphHandlerObj::name(Tensor a, Tensor b, Tensor c) {               \
        if (c) {                                                               \
            g->addOpWithOutputs<obj##Obj>(std::move(a), std::move(b), c);      \
            return c;                                                          \
        } else {                                                               \
            return g->addOp<obj##Obj>(std::move(a), std::move(b), c)           \
                ->getOutput();                                                 \
        }                                                                      \
    }

DEFINE_ELEMENT_WISE_METHOD(add, Add)
DEFINE_ELEMENT_WISE_METHOD(sub, Sub)
DEFINE_ELEMENT_WISE_METHOD(mul, Mul)
DEFINE_ELEMENT_WISE_METHOD(div, Div)
DEFINE_ELEMENT_WISE_METHOD(pow, Pow)

// see operators/unary.h
#define DEFINE_UNARY_METHOD(name, obj)                                         \
    Tensor GraphHandlerObj::name(Tensor x, Tensor y) {                         \
        if (y) {                                                               \
            g->addOpWithOutputs<obj##Obj>(std::move(x), y);                    \
            return y;                                                          \
        } else {                                                               \
            return g->addOp<obj##Obj>(std::move(x), y)->getOutput();           \
        }                                                                      \
    }

DEFINE_UNARY_METHOD(relu, Relu)
DEFINE_UNARY_METHOD(sigmoid, Sigmoid)
DEFINE_UNARY_METHOD(tanh, Tanh)
DEFINE_UNARY_METHOD(abs, Abs)
DEFINE_UNARY_METHOD(shape, Shape)

// see operators/reshape.h
DEFINE_UNARY_METHOD(identity, Identity)

Tensor GraphHandlerObj::pRelu(Tensor x, Tensor slope, Tensor y) {
    if (y) {
        g->addOpWithOutputs<PReluObj>(std::move(x), std::move(slope), y);
        return y;
    } else {
        return g->addOp<PReluObj>(std::move(x), std::move(slope), y)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::clip(Tensor x, Tensor y, std::optional<float> min,
                             std::optional<float> max) {
    if (y) {
        g->addOpWithOutputs<ClipObj>(std::move(x), y, min, max);
        return y;
    } else {
        return g->addOp<ClipObj>(std::move(x), y, min, max)->getOutput();
    }
}

Tensor GraphHandlerObj::softmax(Tensor input, Tensor output, int axis) {
    if (output) {
        g->addOpWithOutputs<SoftmaxObj>(std::move(input), output, axis);
        return output;
    } else {
        return g->addOp<SoftmaxObj>(std::move(input), output, axis)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::flatten(Tensor input, Tensor output, int axis) {
    if (output) {
        g->addOpWithOutputs<FlattenObj>(std::move(input), output, axis);
        return output;
    } else {
        return g->addOp<FlattenObj>(std::move(input), output, axis)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::transpose(Tensor data, Tensor transposed, Shape perm) {
    if (transposed) {
        g->addOpWithOutputs<TransposeObj>(std::move(data), transposed, perm);
        return transposed;
    } else {
        return g->addOp<TransposeObj>(std::move(data), transposed, perm)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::reshape(Tensor data, Tensor reshaped, Shape shape) {
    if (reshaped) {
        g->addOpWithOutputs<ReshapeObj>(std::move(data), reshaped,
                                        std::move(shape));
        return reshaped;
    } else {
        return g->addOp<ReshapeObj>(std::move(data), reshaped, std::move(shape))
            ->getOutput();
    }
}

Tensor GraphHandlerObj::concat(TensorVec inputs, Tensor output, int dim) {
    if (output) {
        g->addOpWithOutputs<ConcatObj>(std::move(inputs), output, dim);
        return output;
    } else {
        return g->addOp<ConcatObj>(std::move(inputs), output, dim)->getOutput();
    }
}

TensorVec GraphHandlerObj::split(Tensor input, std::optional<TensorVec> outputs,
                                 int axis, int num_outputs) {
    if (outputs) {
        g->addOpWithOutputs<SplitObj>(std::move(input), outputs, axis,
                                      num_outputs);
        return *outputs;
    } else {
        return g->addOp<SplitObj>(std::move(input), outputs, axis, num_outputs)
            ->getOutputs();
    }
}

Tensor GraphHandlerObj::gather(Tensor data, Tensor indices, Tensor output,
                               int axis) {
    if (output) {
        g->addOpWithOutputs<GatherObj>(std::move(data), std::move(indices),
                                       output, axis);
        return output;
    } else {
        return g
            ->addOp<GatherObj>(std::move(data), std::move(indices), output,
                               axis)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::reduceMean(Tensor data, Tensor reduced,
                                   const optional<vector<int>> &axes,
                                   bool keepdims) {
    if (reduced) {
        g->addOpWithOutputs<ReduceMeanObj>(std::move(data), reduced, axes,
                                           keepdims);
        return reduced;
    } else {
        return g->addOp<ReduceMeanObj>(std::move(data), reduced, axes, keepdims)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::slice(Tensor input, Tensor output,
                              const vector<int> &starts,
                              const vector<int> &ends,
                              const optional<vector<int>> &axes,
                              const optional<vector<int>> &steps) {
    if (output) {
        g->addOpWithOutputs<SliceObj>(std::move(input), output, starts, ends,
                                      axes, steps);
        return output;
    } else {
        return g
            ->addOp<SliceObj>(std::move(input), output, starts, ends, axes,
                              steps)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::pad(Tensor input, Tensor output,
                            const vector<int> &pads,
                            const optional<vector<int>> &axes) {
    if (output) {
        g->addOpWithOutputs<PadObj>(std::move(input), output, pads, axes);
        return output;
    } else {
        return g->addOp<PadObj>(std::move(input), output, pads, axes)
            ->getOutput();
    }
}

TensorVec GraphHandlerObj::memBound(const TensorVec &inputs,
                                    const Tensor &output,
                                    const string &jsonString) {
    const auto &[expr, nnetInputs, execTime, hint] =
        nnet::Serializer().membundOpFromString(jsonString);
    if (output) {
        g->addOpWithOutputs<MemBoundObj>(std::move(inputs), TensorVec{output},
                                         nnetInputs, expr, execTime, hint);
        return {output};
    } else
        return g
            ->addOp<MemBoundObj>(std::move(inputs), TensorVec{nullptr},
                                 nnetInputs, expr, execTime, hint)
            ->getOutputs();
}

static DataType dtype_repr_convert(int dtype) {
    switch ((OnnxDType)dtype) {
    case OnnxDType::FLOAT:
        return DataType::Float32;
    case OnnxDType::UINT32:
        return DataType::UInt32;
    case OnnxDType::UINT8:
        return DataType::UInt8;
    case OnnxDType::INT8:
        return DataType::Int8;
    case OnnxDType::UINT16:
        return DataType::UInt16;
    case OnnxDType::INT16:
        return DataType::Int16;
    case OnnxDType::INT32:
        return DataType::Int32;
    case OnnxDType::INT64:
        return DataType::Int64;
    default:
        IT_ASSERT(false, "Unsupported data type");
    }
}

Graph GraphHandlerObj::getGraph() const {
    int nRemoved = g->removeIndependentTensors();
    if (nRemoved > 0)
        std::cout << "Removed " << nRemoved << " independent tensors"
                  << std::endl;
    return g;
}

} // namespace infini
