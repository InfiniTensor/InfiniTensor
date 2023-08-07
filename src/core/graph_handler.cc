#include "core/graph_handler.h"
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

Tensor GraphHandlerObj::tensor(Shape dims, int dtype) {
    return g->addTensor(std::move(dims), dtype_repr_convert(dtype));
}

Tensor GraphHandlerObj::conv(Tensor input, Tensor weight, Tensor bias,
                             Tensor output, int ph, int pw, int sh, int sw,
                             int dh, int dw) {
    if (output) {
        g->addOpWithOutputs<ConvObj>(std::move(input), std::move(weight),
                                     output, ph, pw, sh, sw, dh, dw, bias,
                                     ActType::None);
        return output;
    } else {
        return g
            ->addOp<ConvObj>(std::move(input), std::move(weight), output, ph,
                             pw, sh, sw, dh, dw, bias, ActType::None)
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

// see operators/unary.h
#define ADD_UNARY(obj)                                                         \
    if (y) {                                                                   \
        g->addOpWithOutputs<obj##Obj>(std::move(x), y);                        \
        return y;                                                              \
    } else {                                                                   \
        return g->addOp<obj##Obj>(std::move(x), y)->getOutput();               \
    }

Tensor GraphHandlerObj::unary(std::string ty, Tensor x, Tensor y) {
    if (ty == "Abs") {
        ADD_UNARY(Abs)
    } else if (ty == "Acos") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Acosh") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Asin") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Asinh") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Atan") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Atanh") {
        IT_ASSERT_TODO(false);
    } else if (ty == "BitwiseNot") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Ceil") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Cos") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Cosh") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Erf") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Exp") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Floor") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Log") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Neg") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Not") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Relu") {
        ADD_UNARY(Relu)
    } else if (ty == "Round") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Sigmoid") {
        ADD_UNARY(Sigmoid)
    } else if (ty == "Sin") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Sinh") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Sqrt") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Tan") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Tanh") {
        ADD_UNARY(Tanh)
    } else if (ty == "Identity") {
        ADD_UNARY(Identity)
    } else if (ty == "Shape") {
        ADD_UNARY(Shape)
    } else {
        IT_ASSERT(false, "Unsupported unary operator");
    }
}

#undef ADD_UNARY

// see operators/element_wise.h
#define ADD_BINARY(obj)                                                        \
    if (z) {                                                                   \
        g->addOpWithOutputs<obj##Obj>(std::move(x), std::move(y), z);          \
        return y;                                                              \
    } else {                                                                   \
        return g->addOp<obj##Obj>(std::move(x), std::move(y), z)->getOutput(); \
    }

Tensor GraphHandlerObj::binary(std::string ty, Tensor x, Tensor y, Tensor z) {
    if (ty == "Add") {
        ADD_BINARY(Add)
    } else if (ty == "Sub") {
        ADD_BINARY(Sub)
    } else if (ty == "Mul") {
        ADD_BINARY(Mul)
    } else if (ty == "Div") {
        ADD_BINARY(Div)
    } else if (ty == "Pow") {
        ADD_BINARY(Pow)
    } else if (ty == "And") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Or") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Xor") {
        IT_ASSERT_TODO(false);
    } else if (ty == "BitShift") {
        IT_ASSERT_TODO(false);
    } else if (ty == "BitwiseAnd") {
        IT_ASSERT_TODO(false);
    } else if (ty == "BitwiseOr") {
        IT_ASSERT_TODO(false);
    } else if (ty == "BitwiseXor") {
        IT_ASSERT_TODO(false);
    } else if (ty == "Equal") {
        IT_ASSERT_TODO(false);
    } else {
        IT_ASSERT(false, "Unsupported unary operator");
    }
}

#undef ADD_BINARY

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
    default:
        IT_ASSERT(false, "Unsupported data type");
    }
}

} // namespace infini
