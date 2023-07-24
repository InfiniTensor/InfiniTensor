#pragma once
#ifndef OP_TYPE_H
#define OP_TYPE_H

#include "core/tensor.h"
#include <string>
#include <unordered_set>

namespace infini {

struct NewOpType {
    // Clang-format is ambiguous in formating of comment alignment.
    // In order to disambiguate, it is necessary to comment all enum elements.
    enum {
        Abs,                // Unary
        Acos,               // Unary
        Acosh,              // Unary
        Add,                // Binary
        And,                // Binary
        ArgMax,             //
        Asin,               // Binary
        Asinh,              // Binary
        Atan,               // Binary
        Atanh,              // Binary
        AveragePool,        // Pool
        BatchNormalization, //
        Bernoulli,          //
        BitShift,           // Binary
        BitwiseAnd,         // Binary
        BitwiseNot,         // Binary
        BitwiseOr,          // Binary
        BitwiseXor,         // Binary
        BlackmanWindow,     //
        Cast,               // Unary
        CastLike,           //
        Ceil,               // Unary
        Celu,               //
        CenterCropPad,      //
        Clip,               // Unary
        Col2lm,
        Compress,
        Concat,
        ConcatFromSequence,
        ConstantOfShape,
        Conv,
        ConvInteger,
        ConvTranspose,
        Cos,  // Unary
        Cosh, // Unary
        CumSum,
        DFT,
        DeformConv,
        DepthToSpace,
        DequantizeLinear,
        Det,
        Div, // Binary
        Dropout,
        DynamicQuantizeLinear,
        Einsum,
        Elu,
        Equal, // Compair
        Erf,   // Unary
        Exp,   // Unary
        Expand,
        EyeLike,
        Flatten,
        Floor, // Unary
        GRU,
        Gather,
        GatherElements,
        GatherND,
        Gemm,
        GlobalAveragePool, // GlobalPool
        GlobalLpPool,      // GlobalPool
        GlobalMaxPool,     // GlobalPool
        Greater,           // Compair
        GreaterOrEqual,    // Compair
        GridSample,
        GroupNormalization,
        HammingWindow,
        HannWindow,
        HardSigmoid,
        HardSwish,
        Hardmax,
        Identity,
        If,
        InstanceNormalization,
        IsInf,
        IsNaN,
        LRN,
        LSTM,
        LayerNormalization,
        LeakyRelu,
        Less,        // Compair
        LessOrEqual, // Compair
        Log,         // Unary
        LogSoftmax,
        Loop,
        LpNormalization,
        LpPool,
        MatMul,
        MatMulInteger,
        Max,
        MaxPool,
        MaxRoiPool,
        MaxUnpool,
        Mean,
        MeanVarianceNormalization,
        MelWeightMatrix,
        Min,
        Mish,
        Mod, // Binary
        Mul, // Binary
        Multinomial,
        Neg, // Unary
        NegativeLogLikelihoodLoss,
        NonMaxSuppression,
        NonZero,
        Not, // Unary
        OneHot,
        Optional,
        OptionalGetElement,
        OptionalHasElement,
        Or, // Binary
        PRelu,
        Pad,
        Pow, // Binary
        QLinearConv,
        QLinearMatMul,
        QuantizeLinear,
        RNN,
        RandomNormal,
        RandomNormalLike,
        RandomUniform,
        RandomUniformLike,
        Range,
        Reciprocal,
        ReduceL1,        // Reduce
        ReduceL2,        // Reduce
        ReduceLogSum,    // Reduce
        ReduceLogSumExp, // Reduce
        ReduceMax,       // Reduce
        ReduceMean,      // Reduce
        ReduceMin,       // Reduce
        ReduceProd,      // Reduce
        ReduceSum,       // Reduce
        ReduceSumSquare, // Reduce
        Relu,            // Unary
        Reshape,
        Resize,
        ReverseSequence,
        RoiAlign,
        Round, // Unary
        STFT,
        Scan,
        Scatter,
        ScatterElements,
        ScatterND,
        Selu,
        SequenceAt,
        SequenceConstruct,
        SequenceEmpty,
        SequenceErase,
        SequenceInsert,
        SequenceLength,
        SequenceMap,
        Shape,
        Shrink,
        Sigmoid,
        Sign,
        Sin,  // Unary
        Sinh, // Unary
        Size,
        Slice,
        Softmax,
        SoftmaxCrossEntropyLoss,
        Softplus,
        Softsign,
        SpaceToDepth,
        Split,
        SplitToSequence,
        Sqrt,
        Squeeze,
        StringNormalizer,
        Sub,  // Binary
        Sum,  //
        Tan,  // Unary
        Tanh, // unary
        TfIdfVectorizer,
        ThresholdedRelu,
        Tile,
        TopK,
        Transpose,
        Trilu,
        Unique,
        Unsqueeze,
        Upsample,
        Where,
        Xor, // Binary
    } type;

    const char *ToString() const;
    bool isUnary() const;
    bool isBinary() const;
    bool isElementWise() const;
    bool isCompair() const;
    bool isPool() const;
    bool isGlobalPool() const;
};

enum class OpType {
    Unknown = 0,
    // linear
    Conv = 100,
    ConvBackwardFilter,
    ConvBackwardData,
    Matmul,
    ConvTrans,
    ConvTransNHWC,
    G2BMM,
    GBMM,
    Pad,
    Slice,
    Concat,
    Split,
    Transpose,
    Extend,
    MaxPool,
    AvgPool,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Gather,
    ReduceMean,
    Reshape,
    Flatten,
    Identity,
    // element wise
    BatchNorm = 200,
    Softmax,
    Activation,
    Relu,
    ReluBackward,
    PRelu,
    Sigmoid,
    SigmoidBackward,
    Tanh,
    TanhBackward,
    Abs,
    Sin,
    Cos,
    Tan,
    ASin,
    ACos,
    ATan,
    SinH,
    CosH,
    TanH,
    ASinH,
    ACosH,
    ATanH,
    Resize,
    Arange,
    Shape,
    Copy,
    Ceil,
    Floor,
    Clip,
    Erf,
    Exp,
    Fill,
    Log,
    L2Loss,
    Maximum,
    Minimum,
    MSELoss,
    Neg,
    Power,
    Reciprocal,
    Sqrt,
    Rsqrt,
    Cast,
    FloorDiv,
    FloorMod,
    Det,
    Round,
    Square,
    SquaredDifference,
    Hardtanh,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterEqual,
    LessThan,
    LessEqual,
    And,
    Or,
    Xor,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    BitLeftShift,
    BitRightShift,
    Dropout,
    UnaryKernel,
    //
    MemBound = 300,
    MemoryGraph,
};

class OpRegistry {
  public:
    static std::string getOpName(OpType opType) {
#define FOP(op)                                                                \
    case OpType::op:                                                           \
        return #op

        switch (opType) {
            FOP(Unknown);
            // linear
            FOP(Conv);
            FOP(ConvBackwardFilter);
            FOP(ConvBackwardData);
            FOP(Matmul);
            FOP(ConvTrans);
            FOP(G2BMM);
            FOP(GBMM);
            FOP(Pad);
            FOP(Slice);
            FOP(Concat);
            FOP(Split);
            FOP(Transpose);
            FOP(Extend);
            FOP(MaxPool);
            FOP(AvgPool);
            FOP(Add);
            FOP(Sub);
            FOP(Mul);
            FOP(Div);
            FOP(Pow);
            FOP(Gather);
            FOP(ReduceMean);
            FOP(Reshape);
            FOP(Identity);
            FOP(Shape);
            // element wise
            FOP(BatchNorm);
            FOP(Softmax);
            FOP(Activation);
            FOP(Relu);
            FOP(ReluBackward);
            FOP(PRelu);
            FOP(Sigmoid);
            FOP(SigmoidBackward);
            FOP(Tanh);
            FOP(TanhBackward);
            FOP(Abs);
            FOP(Sin);
            FOP(Cos);
            FOP(Tan);
            FOP(ASin);
            FOP(ACos);
            FOP(ATan);
            FOP(SinH);
            FOP(CosH);
            FOP(TanH);
            FOP(ASinH);
            FOP(ACosH);
            FOP(ATanH);
            FOP(Copy);
            FOP(Ceil);
            FOP(Floor);
            FOP(Clip);
            FOP(Erf);
            FOP(Exp);
            FOP(Fill);
            FOP(Log);
            FOP(L2Loss);
            FOP(Maximum);
            FOP(Minimum);
            FOP(MSELoss);
            FOP(Neg);
            FOP(Power);
            FOP(Reciprocal);
            FOP(Sqrt);
            FOP(Rsqrt);
            FOP(Cast);
            FOP(FloorDiv);
            FOP(FloorMod);
            FOP(Det);
            FOP(Round);
            FOP(Square);
            FOP(SquaredDifference);
            FOP(Hardtanh);
            FOP(Equal);
            FOP(NotEqual);
            FOP(GreaterThan);
            FOP(GreaterEqual);
            FOP(LessThan);
            FOP(LessEqual);
            FOP(And);
            FOP(Or);
            FOP(Xor);
            FOP(Not);
            FOP(BitAnd);
            FOP(BitOr);
            FOP(BitXor);
            FOP(BitNot);
            FOP(BitLeftShift);
            FOP(BitRightShift);
            FOP(UnaryKernel);
            //
            FOP(MemBound);
        default:
            IT_ASSERT(false);
            break;
        }
#undef FOP
    }
};

enum class ActType {
    None,
    Relu,
    Sigmoid,
    Tanh,
};

} // namespace infini

#endif // OP_TYPE_H
