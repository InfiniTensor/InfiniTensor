#include "core/graph_handler.h"
#include "operators/all_gather.h"
#include "operators/all_reduce.h"
#include "operators/batch_norm.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/gather.h"
#include "operators/matmul.h"
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
static CastType inferCastType(Tensor input, int to);

Tensor GraphHandlerObj::tensor(Shape dims, int dtype) {
    return g->addTensor(std::move(dims), dtype_repr_convert(dtype));
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

Tensor GraphHandlerObj::batchNormalization(Tensor input, Tensor output,
                                           Tensor mean, Tensor var,
                                           Tensor scale, Tensor bias,
                                           float momentum, float eps,
                                           bool training) {
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

Tensor GraphHandlerObj::allReduceSum(Tensor input, Tensor output) {
    if (output) {
        g->addOpWithOutputs<AllReduceSumObj>(std::move(input), output);
        return output;
    } else {
        return g->addOp<AllReduceSumObj>(std::move(input), output)->getOutput();
    }
}

Tensor GraphHandlerObj::allReduceProd(Tensor input, Tensor output) {
    if (output) {
        g->addOpWithOutputs<AllReduceProdObj>(std::move(input), output);
        return output;
    } else {
        return g->addOp<AllReduceProdObj>(std::move(input), output)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::allReduceMin(Tensor input, Tensor output) {
    if (output) {
        g->addOpWithOutputs<AllReduceMinObj>(std::move(input), output);
        return output;
    } else {
        return g->addOp<AllReduceMinObj>(std::move(input), output)->getOutput();
    }
}

Tensor GraphHandlerObj::allReduceMax(Tensor input, Tensor output) {
    if (output) {
        g->addOpWithOutputs<AllReduceMaxObj>(std::move(input), output);
        return output;
    } else {
        return g->addOp<AllReduceMaxObj>(std::move(input), output)->getOutput();
    }
}

Tensor GraphHandlerObj::allReduceAvg(Tensor input, Tensor output) {
    if (output) {
        g->addOpWithOutputs<AllReduceAvgObj>(std::move(input), output);
        return output;
    } else {
        return g->addOp<AllReduceAvgObj>(std::move(input), output)->getOutput();
    }
}

Tensor GraphHandlerObj::allGather(Tensor input, Tensor output) {
    if (output) {
        g->addOpWithOutputs<AllGatherObj>(std::move(input), output);
        return output;
    } else {
        return g->addOp<AllGatherObj>(std::move(input), output)->getOutput();
    }
}

Tensor GraphHandlerObj::cast(Tensor input, Tensor output, int to) {
    if (output) {
        g->addOpWithOutputs<CastObj>(std::move(input), output,
                                     inferCastType(input, to));
        return output;
    } else {
        return g
            ->addOp<CastObj>(std::move(input), output, inferCastType(input, to))
            ->getOutput();
    }
}

static CastType inferCastType(Tensor input, int to) {
    auto iType = input->getDType();
    auto oType = DataType(to);
    if (iType == DataType::Float32 && oType == DataType::Float16) {
        return CastType::Float2Float16;
    } else if (iType == DataType::Float32 && oType == DataType::Int64) {
        return CastType::Float2Int64;
    } else if (iType == DataType::Float32 && oType == DataType::Int32) {
        return CastType::Float2Int32;
    } else if (iType == DataType::Float32 && oType == DataType::Int16) {
        return CastType::Float2Int16;
    } else if (iType == DataType::Float32 && oType == DataType::Int8) {
        return CastType::Float2Int8;
    } else if (iType == DataType::Float32 && oType == DataType::BFloat16) {
        return CastType::Float2BFloat16;
    } else if (iType == DataType::Int32 && oType == DataType::Float32) {
        return CastType::Int322Float;
    } else if (iType == DataType::Int32 && oType == DataType::Int8) {
        return CastType::Int322Int8;
    } else if (iType == DataType::Int32 && oType == DataType::Int16) {
        return CastType::Int322Int16;
    } else if (iType == DataType::Int32 && oType == DataType::Int64) {
        return CastType::Int322Int64;
    } else if (iType == DataType::Int16 && oType == DataType::Int32) {
        return CastType::Int162Int32;
    } else if (iType == DataType::Int16 && oType == DataType::Float32) {
        return CastType::Int162Float;
    } else if (iType == DataType::Int8 && oType == DataType::Float32) {
        return CastType::Int82Float;
    } else if (iType == DataType::Int8 && oType == DataType::Int16) {
        return CastType::Int82Int16;
    } else if (iType == DataType::Int8 && oType == DataType::Int32) {
        return CastType::Int82Int32;
    } else if (iType == DataType::UInt8 && oType == DataType::Int32) {
        return CastType::Uint82Int32;
    } else if (iType == DataType::UInt8 && oType == DataType::Float32) {
        return CastType::Uint82Float;
    } else if (iType == DataType::UInt8 && oType == DataType::Int64) {
        return CastType::Uint82Int64;
    } else if (iType == DataType::Int64 && oType == DataType::Float32) {
        return CastType::Int642Float;
    } else if (iType == DataType::Int64 && oType == DataType::UInt32) {
        return CastType::Int642Uint32;
    } else if (iType == DataType::Int64 && oType == DataType::Int32) {
        return CastType::Int642Int32;
    } else if (iType == DataType::UInt32 && oType == DataType::Int64) {
        return CastType::Uint322Int64;
    } else if (iType == DataType::Float16 && oType == DataType::Float32) {
        return CastType::Float162Float;
    } else if (iType == DataType::BFloat16 && oType == DataType::Float32) {
        return CastType::BFloat162Float;
    } else {
        IT_TODO_HALT_MSG("Unsupported CastType : input_type is " +
                         iType.toString() + " output_type is " +
                         oType.toString());
    }
}

static DataType dtype_repr_convert(int dtype) {
    switch (dtype) {
    case 0:
        return DataType::Undefine;
    case 1:
        return DataType::Float32;
    case 2:
        return DataType::UInt8;
    case 3:
        return DataType::Int8;
    case 4:
        return DataType::UInt16;
    case 5:
        return DataType::Int16;
    case 6:
        return DataType::Int32;
    case 7:
        return DataType::Int64;
    case 8:
        return DataType::String;
    case 9:
        return DataType::Bool;
    case 10:
        return DataType::Float16;
    case 11:
        return DataType::Double;
    case 12:
        return DataType::UInt32;
    case 13:
        return DataType::UInt64;
    case 16:
        return DataType::BFloat16;
    default:
        IT_ASSERT(false, "Unsupported data type");
    }
}

} // namespace infini
