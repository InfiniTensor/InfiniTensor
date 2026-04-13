#include "core/graph_handler.h"
#include "operators/all_gather.h"
#include "operators/all_reduce.h"
#include "operators/attention_kvcache.h"
#include "operators/batch_norm.h"
#include "operators/broadcast.h"
#include "operators/concat.h"
#include "operators/conv.h"
#include "operators/element_wise.h"
#include "operators/expand.h"
#include "operators/gather.h"
#include "operators/instance_norm.h"
#include "operators/layer_norm.h"
#include "operators/lrn.h"
#include "operators/matmul.h"
#include "operators/pad.h"
#include "operators/pooling.h"
#include "operators/recv.h"
#include "operators/reduce.h"
#include "operators/reshape.h"
#include "operators/resize.h"
#include "operators/rms_norm.h"
#include "operators/rope.h"
#include "operators/send.h"
#include "operators/slice.h"
#include "operators/softmax.h"
#include "operators/split.h"
#include "operators/squeeze.h"
#include "operators/transpose.h"
#include "operators/unary.h"
#include "operators/unsqueeze.h"
#include "operators/where.h"
#include "operators/argmax.h"
#include <numeric>

namespace infini {

static DataType dtype_repr_convert(int);
static CastType inferCastType(Tensor input, int to);

Tensor GraphHandlerObj::elu(Tensor input, Tensor output, float alpha) {
    if (output) {
        g->addOpWithOutputs<EluObj>(std::move(input), output, alpha);
        return output;
    } else {
        auto new_output = g->addTensor(input->getDims(), input->getDType());
        g->addOpWithOutputs<EluObj>(std::move(input), new_output, alpha);
        return new_output;
    }
}

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
                               bool transB, Tensor bias, ActType act,
                               std::string matmul_compute_type) {
    if (y) {
        g->addOpWithOutputs<MatmulObj>(std::move(a), std::move(b), y, transA,
                                       transB, std::move(bias), act,
                                       matmul_compute_type);
        return y;
    } else {
        return g
            ->addOp<MatmulObj>(std::move(a), std::move(b), y, transA, transB,
                               std::move(bias), act, matmul_compute_type)
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

Tensor GraphHandlerObj::layerNormalization(Tensor input, Tensor scale,
                                           Tensor output, Tensor bias,
                                           float eps, int axis,
                                           int stash_type) {
    if (output) {
        g->addOpWithOutputs<LayerNormObj>(std::move(input), std::move(scale),
                                          output, std::move(bias), eps, axis,
                                          stash_type);
        return output;
    } else {
        return g
            ->addOp<LayerNormObj>(std::move(input), std::move(scale), output,
                                  std::move(bias), eps, axis, stash_type)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::instanceNormalization(Tensor input, Tensor output,
                                              Tensor scale, Tensor bias,
                                              float eps) {
    if (output) {
        g->addOpWithOutputs<InstanceNormObj>(
            std::move(input), output, std::move(scale), std::move(bias), eps);
        return output;
    } else {
        return g
            ->addOp<InstanceNormObj>(std::move(input), output, std::move(scale),
                                     std::move(bias), eps)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::rmsNorm(Tensor input, Tensor weight, Tensor output) {
    if (output) {
        g->addOpWithOutputs<RMSNormObj>(std::move(input), std::move(weight),
                                        output);
        return output;
    } else {
        return g->addOp<RMSNormObj>(std::move(input), std::move(weight), output)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::maxPool(Tensor input, Tensor output, int kh, int kw,
                                int dh, int dw, int ph, int pw, int sh, int sw,
                                int ceilMode) {
    if (output) {
        g->addOpWithOutputs<MaxPoolObj>(std::move(input), output, kh, kw, dh,
                                        dw, ph, pw, sh, sw, ceilMode);
        return output;
    } else {
        return g
            ->addOp<MaxPoolObj>(std::move(input), output, kh, kw, dh, dw, ph,
                                pw, sh, sw, ceilMode)
            ->getOutput();
    }
}
Tensor GraphHandlerObj::avgPool(Tensor input, Tensor output, int kh, int kw,
                                int dh, int dw, int ph, int pw, int sh, int sw,
                                int ceilMode) {
    if (output) {
        g->addOpWithOutputs<AvgPoolObj>(std::move(input), output, kh, kw, dh,
                                        dw, ph, pw, sh, sw, ceilMode);
        return output;
    } else {
        return g
            ->addOp<AvgPoolObj>(std::move(input), output, kh, kw, dh, dw, ph,
                                pw, sh, sw, ceilMode)
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
DEFINE_ELEMENT_WISE_METHOD(min, Minimum)
DEFINE_ELEMENT_WISE_METHOD(max, Maximum)

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

DEFINE_UNARY_METHOD(silu, Silu)
DEFINE_UNARY_METHOD(relu, Relu)
DEFINE_UNARY_METHOD(gelu, Gelu)
DEFINE_UNARY_METHOD(sigmoid, Sigmoid)
DEFINE_UNARY_METHOD(tanh, Tanh)
DEFINE_UNARY_METHOD(hardSigmoid, HardSigmoid)
DEFINE_UNARY_METHOD(hardSwish, HardSwish)
DEFINE_UNARY_METHOD(abs, Abs)
DEFINE_UNARY_METHOD(sqrt, Sqrt)
DEFINE_UNARY_METHOD(neg, Neg)
DEFINE_UNARY_METHOD(shape, Shape)
DEFINE_UNARY_METHOD(erf, Erf)

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

Tensor GraphHandlerObj::leakyRelu(Tensor x, Tensor y, float alpha) {
    if (y) {
        g->addOpWithOutputs<LeakyReluObj>(std::move(x), y, alpha);
        return y;
    } else {
        return g->addOp<LeakyReluObj>(std::move(x), y, alpha)->getOutput();
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

Tensor GraphHandlerObj::resize(Tensor input, Tensor output,
                               const std::optional<vector<int>> &axes,
                               Tensor sizes, Tensor scales, Tensor roi,
                               vector<int64_t> sizes_, vector<float> scales_,
                               vector<float> roi_, string mode,
                               string ratioPolicy, string nearestMode,
                               string coordTransMode) {
    if (sizes_.size() > 0) {
        sizes->dataMalloc();
        sizes->copyin<int64_t>(sizes_);
    }
    if (scales_.size() > 0) {
        scales->dataMalloc();
        scales->copyin<float>(scales_);
    }
    if (roi_.size() > 0) {
        roi->dataMalloc();
        roi->copyin<float>(roi_);
    }
    ResizeObj::EKeepAspectRatioPolicy ratioPolicy_ =
        ResizeObj::fromRatioPolicyStr(ratioPolicy);
    ResizeObj::ENearestMode nearestMode_ =
        ResizeObj::fromENearestModeStr(nearestMode);
    ResizeObj::ECoordinateTransMode coordTransMode_ =
        ResizeObj::fromECoordinateTransModeStr(coordTransMode);
    ResizeObj::ECoeffMode mode_ = ResizeObj::fromECoeffModeStr(mode);
    if (output) {
        if (mode == "nearest") {
            g->addOpWithOutputs<ResizeObj>(
                std::move(input), output, std::move(axes), std::move(sizes),
                std::move(scales), std::move(roi), ratioPolicy_, nearestMode_,
                coordTransMode_);
        } else {
            g->addOpWithOutputs<ResizeObj>(
                std::move(input), output, std::move(axes), std::move(sizes),
                std::move(scales), std::move(roi), mode_, ratioPolicy_,
                coordTransMode_);
        }
        return output;
    } else {
        if (mode == "nearest") {
            return g
                ->addOp<ResizeObj>(std::move(input), output, std::move(axes),
                                   std::move(sizes), std::move(scales),
                                   std::move(roi), ratioPolicy_, nearestMode_,
                                   coordTransMode_)
                ->getOutput();
        } else {
            return g
                ->addOp<ResizeObj>(std::move(input), output, std::move(axes),
                                   std::move(sizes), std::move(scales),
                                   std::move(roi), mode_, ratioPolicy_,
                                   coordTransMode_)
                ->getOutput();
        }
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

Tensor GraphHandlerObj::attentionKVCache(Tensor input_k_cache,
                                         Tensor input_v_cache, Tensor input_q,
                                         Tensor input_k, Tensor input_v,
                                         Tensor position_id,
                                         Tensor output_matmul) {
    if (output_matmul) {
        g->addOpWithOutputs<AttentionKVCacheObj>(
            std::move(input_k_cache), std::move(input_v_cache),
            std::move(input_q), std::move(input_k), std::move(input_v),
            std::move(position_id), output_matmul);
        return output_matmul;
    } else {
        return g
            ->addOp<AttentionKVCacheObj>(
                std::move(input_k_cache), std::move(input_v_cache),
                std::move(input_q), std::move(input_k), std::move(input_v),
                std::move(position_id), output_matmul)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::RoPE(Tensor pos, Tensor input, Tensor output) {
    if (output) {
        g->addOpWithOutputs<RoPEObj>(std::move(pos), std::move(input), output);
        return output;
    } else {
        return g->addOp<RoPEObj>(std::move(pos), std::move(input), output)
            ->getOutput();
    }
}
// TODO: implement argmax
Tensor GraphHandlerObj::argmax(Tensor input, Tensor output, int axis, bool keepdim) {
    if (output) {
        g->addOpWithOutputs<ArgMaxObj>(std::move(input), output, axis);
        return output;
    } else {
        return g->addOp<ArgMaxObj>(std::move(input), output, axis)
            ->getOutput();
    }
}

TensorVec GraphHandlerObj::split(Tensor input, std::optional<TensorVec> outputs,
                                 int axis,
                                 std::variant<int, vector<int>> numOrRatio) {
    if (outputs) {
        if (std::holds_alternative<int>(numOrRatio)) {
            g->addOpWithOutputs<SplitObj>(std::move(input), outputs, axis,
                                          std::get<int>(numOrRatio));
        } else {
            g->addOpWithOutputs<SplitObj>(std::move(input), outputs, axis,
                                          std::get<vector<int>>(numOrRatio));
        }
        return *outputs;
    } else {
        if (std::holds_alternative<int>(numOrRatio)) {
            return g
                ->addOp<SplitObj>(std::move(input), outputs, axis,
                                  std::get<int>(numOrRatio))
                ->getOutputs();
        } else {
            return g
                ->addOp<SplitObj>(std::move(input), outputs, axis,
                                  std::get<vector<int>>(numOrRatio))
                ->getOutputs();
        }
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

Tensor GraphHandlerObj::gatherElements(Tensor data, Tensor indices,
                                       Tensor output, int axis) {
    if (output) {
        g->addOpWithOutputs<GatherElementsObj>(
            std::move(data), std::move(indices), output, axis);
        return output;
    } else {
        return g
            ->addOp<GatherElementsObj>(std::move(data), std::move(indices),
                                       output, axis)
            ->getOutput();
    }
}

#define DEFINE_REDUCE_METHOD(name, obj)                                        \
    Tensor GraphHandlerObj::name(Tensor data, Tensor reduced,                  \
                                 const optional<vector<int>> &axes,            \
                                 bool keepdims) {                              \
        if (reduced) {                                                         \
            g->addOpWithOutputs<_CAT(obj, Obj)>(std::move(data), reduced,      \
                                                axes, keepdims);               \
            return reduced;                                                    \
        } else {                                                               \
            return g                                                           \
                ->addOp<_CAT(obj, Obj)>(std::move(data), reduced, axes,        \
                                        keepdims)                              \
                ->getOutput();                                                 \
        }                                                                      \
    }
DEFINE_REDUCE_METHOD(reduceMean, ReduceMean)
DEFINE_REDUCE_METHOD(reduceSum, ReduceSum)

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

TensorVec GraphHandlerObj::allGather(Tensor input,
                                     std::optional<TensorVec> outputs, int n) {
    if (outputs) {
        g->addOpWithOutputs<AllGatherObj>(std::move(input), outputs, n);
        return *outputs;
    } else {
        return g->addOp<AllGatherObj>(std::move(input), outputs, n)
            ->getOutputs();
    }
}

Tensor GraphHandlerObj::broadcast(Tensor input, Tensor output, int root) {
    if (output) {
        g->addOpWithOutputs<BroadcastObj>(std::move(input), output, root);
        return output;
    } else {
        return g->addOp<BroadcastObj>(std::move(input), output, root)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::send(Tensor input, int source, int destination,
                             Tensor output) {
    if (output) {

        g->addOpWithOutputs<SendObj>(std::move(input), source, destination,
                                     output);

        return output;
    } else {
        return g->addOp<SendObj>(std::move(input), source, destination, output)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::recv(Tensor output, int source, int destination,
                             Shape dims, int outputType, Tensor input) {

    if (output) {

        g->addOpWithOutputs<RecvObj>(output, source, destination,
                                     std::move(dims), outputType,
                                     std::move(input));

        return output;
    } else {

        return g
            ->addOp<RecvObj>(output, source, destination, std::move(dims),
                             outputType, std::move(input))
            ->getOutput();
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

Tensor GraphHandlerObj::expand(Tensor input, Tensor output, Shape dims) {
    if (output) {
        g->addOpWithOutputs<ExpandObj>(std::move(input), output,
                                       std::move(dims));
        return output;
    } else {
        return g->addOp<ExpandObj>(std::move(input), output, std::move(dims))
            ->getOutput();
    }
}

Tensor GraphHandlerObj::where(Tensor inputX, Tensor inputY, Tensor condition,
                              Tensor output) {
    if (output) {
        g->addOpWithOutputs<WhereObj>(std::move(inputX), std::move(inputY),
                                      std::move(condition), output);
        return output;
    } else {
        return g
            ->addOp<WhereObj>(std::move(inputX), std::move(inputY),
                              std::move(condition), output)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::depthToSpace(Tensor input, Tensor output, int blocksize,
                                     std::string mode) {
    if (output) {
        g->addOpWithOutputs<DepthToSpaceObj>(std::move(input), output,
                                             blocksize, mode);
        return output;
    } else {
        return g
            ->addOp<DepthToSpaceObj>(std::move(input), output, blocksize, mode)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::lrn(Tensor input, Tensor output, float alpha,
                            float beta, float bias, int size) {
    if (output) {
        g->addOpWithOutputs<LRNObj>(std::move(input), output, alpha, beta, bias,
                                    size);
        return output;
    } else {
        return g
            ->addOp<LRNObj>(std::move(input), output, alpha, beta, bias, size)
            ->getOutput();
    }
}

Tensor GraphHandlerObj::squeeze(Tensor input, Tensor output, Shape axes) {
    if (output) {
        g->addOpWithOutputs<SqueezeObj>(std::move(input), output,
                                        std::move(axes));
        return output;
    } else {
        return g->addOp<SqueezeObj>(std::move(input), output, std::move(axes))
            ->getOutput();
    }
}

Tensor GraphHandlerObj::unsqueeze(Tensor input, Tensor output, Shape axes) {
    if (output) {
        g->addOpWithOutputs<UnsqueezeObj>(std::move(input), output,
                                          std::move(axes));
        return output;
    } else {
        return g->addOp<UnsqueezeObj>(std::move(input), output, std::move(axes))
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
    } else if (iType == DataType::Float32 && oType == DataType::Float32) {
        return CastType::Float2Float;
    } else if (iType == DataType::Float32 && oType == DataType::Bool) {
        return CastType::Float322Bool;
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

void GraphHandlerObj::change_shape(const vector<int> &shape, int tensorId) {
    auto tensor = g->getTensor(tensorId);
    IT_ASSERT(tensor != nullptr);
    IT_ASSERT(shape.size() != 0);
    tensor->setShape(shape);
}

} // namespace infini
