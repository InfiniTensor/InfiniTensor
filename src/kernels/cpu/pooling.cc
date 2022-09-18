#include "operators/pooling.h"
#include "core/kernel.h"

namespace infini {
template <typename T> class NativePooling : public CpuKernelWithoutConfig {
    virtual T getPoolingValue(int kh, int kw, int posh, int posw, int ih,
                              int iw, T *inptr) const = 0;
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<PoolingObj>(_op);
        T *inptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *outptr = op->getOutput()->getRawDataPtr<T *>();
        const auto [n, c, ih, iw, kh, kw] = op->getNCHWRS();
        const auto [ph, pw, sh, sw, dh, dw] = op->getPadStrideDilation();
        if (dh != 1 || dw != 1)
            IT_TODO_HALT(); // To support dailated pooling
        auto outDim = op->getOutput()->getDims();
        int oh = outDim[2], ow = outDim[3];
        for (auto i = 0; i < n; i++) {
            for (auto j = 0; j < c; j++) {
                auto inoffset = i * (c * ih * iw) + j * ih * iw;
                for (auto h = 0; h < oh; h++) {
                    for (auto w = 0; w < ow; w++) {
                        T val =
                            getPoolingValue(kh, kw, h * sh - ph, w * sw - pw,
                                            ih, iw, inptr + inoffset);
                        auto outoffset =
                            w + h * ow + j * (oh * ow) + i * (c * oh * ow);
                        outptr[outoffset] = val;
                    }
                }
            }
        }
    }
};

template <typename T> class NaiveMaxPool : public NativePooling<T> {
    T getPoolingValue(int kh, int kw, int posh, int posw, int ih, int iw,
                      T *inptr) const override {
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
};

template <typename T> class NaiveAvgPool : public NativePooling<T> {
    T getPoolingValue(int kh, int kw, int posh, int posw, int ih, int iw,
                      T *inptr) const override {
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
};

REGISTER_KERNEL(Device::CPU, OpType::MaxPool, DataType::UInt32,
                NaiveMaxPool<uint32_t>, "maxPoolNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::MaxPool, DataType::Float32,
                NaiveMaxPool<float>, "maxPoolNaive_CPU_float32");
REGISTER_KERNEL(Device::CPU, OpType::AvgPool, DataType::Float32,
                NaiveAvgPool<float>, "AvgPoolNaive_CPU_float32");
} // namespace infini