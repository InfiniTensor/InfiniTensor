#include "core/op_type.h"

namespace infini {
const char *OpType::toString() const {
#define CASE(NAME)                                                             \
    case OpType::NAME:                                                         \
        return #NAME

    switch (type) {
        CASE(Unknown);
        CASE(Abs);
        CASE(Acos);
        CASE(Acosh);
        CASE(Add);
        CASE(And);
        CASE(ArgMax);
        CASE(Asin);
        CASE(Asinh);
        CASE(Atan);
        CASE(Atanh);
        CASE(AveragePool);
        CASE(BatchNormalization);
        CASE(Bernoulli);
        CASE(BitShift);
        CASE(BitwiseAnd);
        CASE(BitwiseNot);
        CASE(BitwiseOr);
        CASE(BitwiseXor);
        CASE(BlackmanWindow);
        CASE(Cast);
        CASE(CastLike);
        CASE(Ceil);
        CASE(Celu);
        CASE(CenterCropPad);
        CASE(Clip);
        CASE(Col2lm);
        CASE(Compress);
        CASE(Concat);
        CASE(ConcatFromSequence);
        CASE(ConstantOfShape);
        CASE(Conv);
        CASE(Conv3d);
        CASE(ConvInteger);
        CASE(ConvTranspose);
        CASE(Cos);
        CASE(Cosh);
        CASE(CumSum);
        CASE(DFT);
        CASE(DeformConv);
        CASE(DepthToSpace);
        CASE(DequantizeLinear);
        CASE(Det);
        CASE(Div);
        CASE(Dropout);
        CASE(DynamicQuantizeLinear);
        CASE(Einsum);
        CASE(Elu);
        CASE(Equal);
        CASE(Erf);
        CASE(Exp);
        CASE(Expand);
        CASE(EyeLike);
        CASE(Flatten);
        CASE(Floor);
        CASE(GRU);
        CASE(Gather);
        CASE(GatherElements);
        CASE(GatherND);
        CASE(Gemm);
        CASE(GlobalAveragePool);
        CASE(GlobalLpPool);
        CASE(GlobalMaxPool);
        CASE(Greater);
        CASE(GreaterOrEqual);
        CASE(GridSample);
        CASE(GroupNormalization);
        CASE(HammingWindow);
        CASE(HannWindow);
        CASE(HardSigmoid);
        CASE(HardSwish);
        CASE(Hardmax);
        CASE(Identity);
        CASE(If);
        CASE(InstanceNormalization);
        CASE(IsInf);
        CASE(IsNaN);
        CASE(LRN);
        CASE(LSTM);
        CASE(LayerNormalization);
        CASE(LeakyRelu);
        CASE(Less);
        CASE(LessOrEqual);
        CASE(Log);
        CASE(LogSoftmax);
        CASE(Loop);
        CASE(LpNormalization);
        CASE(LpPool);
        CASE(MatMul);
        CASE(MatMulInteger);
        CASE(Max);
        CASE(MaxPool);
        CASE(MaxRoiPool);
        CASE(MaxUnpool);
        CASE(Mean);
        CASE(MeanVarianceNormalization);
        CASE(MelWeightMatrix);
        CASE(Min);
        CASE(Mish);
        CASE(Mod);
        CASE(Mul);
        CASE(Multinomial);
        CASE(Neg);
        CASE(NegativeLogLikelihoodLoss);
        CASE(NonMaxSuppression);
        CASE(NonZero);
        CASE(Not);
        CASE(OneHot);
        CASE(Optional);
        CASE(OptionalGetElement);
        CASE(OptionalHasElement);
        CASE(Or);
        CASE(PRelu);
        CASE(Pad);
        CASE(Pow);
        CASE(QLinearConv);
        CASE(QLinearMatMul);
        CASE(QuantizeLinear);
        CASE(RNN);
        CASE(RandomNormal);
        CASE(RandomNormalLike);
        CASE(RandomUniform);
        CASE(RandomUniformLike);
        CASE(Range);
        CASE(Reciprocal);
        CASE(ReduceL1);
        CASE(ReduceL2);
        CASE(ReduceLogSum);
        CASE(ReduceLogSumExp);
        CASE(ReduceMax);
        CASE(ReduceMean);
        CASE(ReduceMin);
        CASE(ReduceProd);
        CASE(ReduceSum);
        CASE(ReduceSumSquare);
        CASE(Relu);
        CASE(Gelu);
        CASE(Reshape);
        CASE(Resize);
        CASE(ReverseSequence);
        CASE(RoiAlign);
        CASE(Round);
        CASE(STFT);
        CASE(Scan);
        CASE(Scatter);
        CASE(ScatterElements);
        CASE(ScatterND);
        CASE(Selu);
        CASE(SequenceAt);
        CASE(SequenceConstruct);
        CASE(SequenceEmpty);
        CASE(SequenceErase);
        CASE(SequenceInsert);
        CASE(SequenceLength);
        CASE(SequenceMap);
        CASE(Shape);
        CASE(Shrink);
        CASE(Sigmoid);
        CASE(Sign);
        CASE(Sin);
        CASE(Sinh);
        CASE(Size);
        CASE(Slice);
        CASE(Softmax);
        CASE(SoftmaxCrossEntropyLoss);
        CASE(Softplus);
        CASE(Softsign);
        CASE(SpaceToDepth);
        CASE(Split);
        CASE(SplitToSequence);
        CASE(Sqrt);
        CASE(Squeeze);
        CASE(StringNormalizer);
        CASE(Sub);
        CASE(Sum);
        CASE(Tan);
        CASE(Tanh);
        CASE(TfIdfVectorizer);
        CASE(ThresholdedRelu);
        CASE(Tile);
        CASE(TopK);
        CASE(Transpose);
        CASE(Trilu);
        CASE(Unique);
        CASE(Unsqueeze);
        CASE(Upsample);
        CASE(Where);
        CASE(Xor);
        // CUSTOM DEFINED
        CASE(G2BMM);
        CASE(GBMM);
        CASE(MemBound);
        // TODO
        CASE(ConvTransNHWC);
        CASE(ConvBackwardFilter);
        CASE(ReluBackward);
        CASE(SigmoidBackward);
        CASE(TanhBackward);

        CASE(Fill);
        CASE(Extend);
        CASE(MSELoss);
        CASE(Hardtanh);
        CASE(L2Loss);
        CASE(Rsqrt);
        CASE(FloorDiv);
        CASE(FloorMod);
        CASE(Square);
        CASE(SquaredDifference);

        // Communcation
        CASE(AllReduceSum);
        CASE(AllReduceProd);
        CASE(AllReduceMin);
        CASE(AllReduceMax);
        CASE(AllReduceAvg);
        CASE(AllGather);
        CASE(Broadcast);
    default:
        return "Unknown";
    }

#undef CASE
}

bool OpType::isUnary() const {
    static const std::unordered_set<decltype(type)> set{
        Abs,  Acos, Acosh, Asin,    Asinh, Atan,  Atanh, Cast, Ceil,
        Clip, Cos,  Cosh,  Erf,     Exp,   Floor, Log,   Neg,  Not,
        Relu, Gelu, Round, Sigmoid, Sin,   Sinh,  Sqrt,  Tan,  Tanh,
    };

    return set.find(type) != set.end();
}

bool OpType::isBinary() const {
    static const std::unordered_set<decltype(type)> set{
        Add, And, BitShift, BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor,
        Div, Mod, Mul,      Or,         Pow,        Sub,       Xor,
    };

    return set.find(type) != set.end() || isCompair();
}

bool OpType::isElementWise() const { return isUnary() || isBinary(); }

bool OpType::isCompair() const {
    static const std::unordered_set<decltype(type)> set{
        Equal, Greater, GreaterOrEqual, Less, LessOrEqual,
    };

    return set.find(type) != set.end();
}

bool OpType::isPool() const {
    static const std::unordered_set<decltype(type)> set{};

    return set.find(type) != set.end();
}

bool OpType::isGlobalPool() const {
    static const std::unordered_set<decltype(type)> set{
        GlobalAveragePool,
        GlobalLpPool,
        GlobalMaxPool,
    };

    return set.find(type) != set.end();
}

bool OpType::isMatMulOrConv() const {
    static const std::unordered_set<decltype(type)> set{
        Conv,        Conv3d, ConvInteger,   ConvTranspose, DeformConv,
        QLinearConv, MatMul, MatMulInteger, QLinearMatMul,
    };

    return set.find(type) != set.end();
}

} // namespace infini
