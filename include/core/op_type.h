#pragma once
#ifndef OP_TYPE_H
#define OP_TYPE_H

#include <cstdint>
#include <string>
#include <unordered_set>

namespace infini {

struct OpType {
    using underlying_t = uint16_t;

    // Clang-format is ambiguous in formating of comment alignment.
    // In order to disambiguate, it is necessary to comment all enum
    // elements.
    enum : underlying_t {
        Unknown,
        Abs,                // Unary
        Acos,               // Unary
        Acosh,              // Unary
        Add,                // Binary
        And,                // Binary
        ArgMax,             //
        Asin,               // Unary
        Asinh,              // Unary
        Atan,               // Unary
        Atanh,              // Unary
        AttentionKVCache,   // Fusion
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
        Conv,          // ComputationIntensive
        Conv3d,        // ComputationIntensive
        ConvInteger,   // ComputationIntensive
        ConvTranspose, // ComputationIntensive
        Cos,           // Unary
        Cosh,          // Unary
        CumSum,
        DFT,
        DeformConv, // ComputationIntensive
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
        Gelu,              // Unary
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
        MatMul,        // ComputationIntensive
        MatMulInteger, // ComputationIntensive
        Max,
        MaxPool,
        MaxRoiPool,
        MaxUnpool,
        Mean,
        MeanVarianceNormalization,
        MelWeightMatrix,
        Min,
        Mish,
        Mod,         // Binary
        Mul,         // Binary
        Multinomial, //
        Neg,         // Unary
        NegativeLogLikelihoodLoss,
        NonMaxSuppression,
        NonZero,
        Not, // Unary
        OneHot,
        Optional,
        OptionalGetElement,
        OptionalHasElement,
        Or,            // Binary
        PRelu,         //
        Pad,           //
        Pow,           // Binary
        QLinearConv,   // ComputationIntensive
        QLinearMatMul, // ComputationIntensive
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
        Silu,            // Unary
        Reshape,
        Resize,
        ReverseSequence,
        RoiAlign,
        RoPE,    // Fusion
        Round,   // Unary
        RMSNorm, // Fusion
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
        // CUSTOM DEFINED
        G2BMM,
        GBMM,
        MemBound,
        // TODO
        ConvTransNHWC,
        ConvBackwardFilter,
        ReluBackward,
        SigmoidBackward,
        TanhBackward,

        Fill,
        Extend,
        MSELoss,
        Hardtanh,
        L2Loss,
        Rsqrt,
        FloorDiv,
        FloorMod,
        Square,
        SquaredDifference,

        // Communication Ops
        AllReduceSum,
        AllReduceProd,
        AllReduceMin,
        AllReduceMax,
        AllReduceAvg,
        AllGather,
        Broadcast,
        Send,
        Recv,
    } type;

    constexpr OpType(decltype(type) t) : type(t) {}
    constexpr explicit OpType(underlying_t val) : type((decltype(type))val) {}
    constexpr underlying_t underlying() const { return type; }

    bool operator==(OpType others) const { return type == others.type; }
    bool operator!=(OpType others) const { return type != others.type; }
    bool operator<(OpType others) const { return type < others.type; }

    const char *toString() const;
    bool isUnary() const;
    bool isBinary() const;
    bool isElementWise() const;
    bool isCompair() const;
    bool isPool() const;
    bool isGlobalPool() const;
    bool isMatMulOrConv() const;
};

enum class ActType {
    None,
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
};

} // namespace infini

#endif // OP_TYPE_H
