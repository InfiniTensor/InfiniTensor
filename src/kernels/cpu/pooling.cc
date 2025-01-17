#include "operators/pooling.h"
#include "core/kernel.h"

namespace infini {
class NativePooling : public CpuKernelWithoutConfig {
    template <typename T>
    static T getMaxPoolingValue(int kh, int kw, int posh, int posw, int ih,
                                int iw, T *inptr) {
        T maxval = 0;
        for (auto k = 0; k < kh; k++) {
            for (auto l = 0; l < kw; l++) {
                auto inPosH = posh + k;
                auto inPosW = posw + l;
                if (inPosH < 0 || inPosH >= ih || inPosW < 0 || inPosW >= iw)
                    continue;
                auto offset = (posh + k) * iw + posw + l;
                auto val = inptr[offset];
                if (maxval < val)
                    maxval = val;
            }
        }
        return maxval;
    }

    template <typename T>
    static T getAvgPoolingValue(int kh, int kw, int posh, int posw, int ih,
                                int iw, T *inptr) {
        T sum = 0;
        for (auto k = 0; k < kh; k++) {
            for (auto l = 0; l < kw; l++) {
                auto inPosH = posh + k;
                auto inPosW = posw + l;
                if (inPosH < 0 || inPosH >= ih || inPosW < 0 || inPosW >= iw)
                    continue;
                auto offset = (posh + k) * iw + posw + l;
                sum += inptr[offset];
            }
        }
        return T(sum / (kh * kw));
    }

    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<PoolingObj>(_op);
        T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();

        const auto [n, c, ih, iw, kh, kw] = op->getNCHWRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        if (dh != 1 || dw != 1)
            IT_TODO_HALT(); // To support dailated pooling
        auto outDim = op->getOutput()->getDims();
        int oh = outDim[2], ow = outDim[3];

        T(*_doCompute)
        (int kh, int kw, int posh, int posw, int ih, int iw, T *inptr);
        switch (op->getOpType().underlying()) {
        case OpType::MaxPool:
            _doCompute = getMaxPoolingValue<T>;
            break;
        case OpType::AveragePool:
            _doCompute = getAvgPoolingValue<T>;
            break;
        default:
            IT_TODO_HALT();
        }

        for (auto i = 0; i < n; i++) {
            for (auto j = 0; j < c; j++) {
                auto inoffset = i * (c * ih * iw) + j * ih * iw;
                for (auto h = 0; h < oh; h++) {
                    for (auto w = 0; w < ow; w++) {
                        // TODO: verify ceil mode
                        T val = _doCompute(kh, kw, h * sh - ph, w * sw - pw, ih,
                                           iw, inptr + inoffset);
                        auto outoffset =
                            w + h * ow + j * (oh * ow) + i * (c * oh * ow);
                        outptr[outoffset] = val;
                    }
                }
            }
        }
    }

    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
#define CASE(N)                                                                \
    case N:                                                                    \
        doCompute<DT<N>::t>(_op, context)

        int dataTypeIdx = _op->getDType().getIndex();
        switch (dataTypeIdx) {
            CASE(1); // DataType::Float32
            break;
            CASE(12); // DataType::UInt32
            break;
        default:
            IT_TODO_HALT();
        }
    }
};

// REGISTER_KERNEL(Device::CPU, OpType::MaxPool, NativePooling,
//                 "maxPoolNaive_CPU");
// REGISTER_KERNEL(Device::CPU, OpType::AveragePool, NativePooling,
//                 "avgPoolNaive_CPU");
} // namespace infini
