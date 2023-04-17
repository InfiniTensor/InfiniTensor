#include "operators/conv.h"
#include "core/kernel.h"

namespace infini {

template <typename T> class NaiveConv : public CpuKernelWithoutConfig {
    void compute(const Operator &_op,
                 const RuntimeObj *context) const override {
        auto op = as<ConvObj>(_op);
        T *iptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *wptr = op->getInputs(1)->getRawDataPtr<T *>();
        T *optr = op->getOutput()->getRawDataPtr<T *>();
        //  Clang will give an error of " reference to local binding 'sh'
        //  declared in enclosing function" if we write like this:
        //        auto [n, c, h, w, f, r, s] = op->getNCHWFRS();
        int n, c, h, w, f, r, s;
        std::tie(n, c, h, w, f, r, s) = op->getNCHWFRS();
        int ph, pw, sh, sw, dh, dw;
        std::tie(ph, pw, sh, sw, dh, dw) = op->getPadStrideDilation();
        int cpg = op->getChannelPerGroup();
        int g = op->getNumGroups();
        IT_ASSERT(f % g == 0, "Illegal number of channel");
        auto outDim = op->getOutput()->getDims();
        int oh = outDim[2], ow = outDim[3];
        for (int nn = 0; nn < n; nn++) {
#pragma omp parallel for
            for (int ff = 0; ff < f; ff++) {
                for (int hh = 0; hh < oh; hh++)
                    for (int ww = 0; ww < ow; ww++) {
                        int gidx = ff / (f / g);
                        T val = 0;
                        for (int cc = 0; cc < cpg; cc++)
                            for (int rr = 0; rr < r; rr++)
                                for (int ss = 0; ss < s; ss++) {
                                    // clang-format off
                                int posH = hh * sh + rr * dh - ph;
                                int posW = ww * sw + ss * dw - pw;
                                if (posH < 0 || posH >= h || posW < 0 || posW >= w)
                                    continue;
                                auto iOffset = posW + w * (posH + h * ((cc + gidx * cpg) + c * nn)),
                                    wOffset = ss + s * (rr + r * (cc + cpg * ff));
                                auto inputVal = iptr[iOffset], weightVal = wptr[wOffset];
                                val += weightVal * inputVal;
                                    // clang-format on
                                }
                        // TODO: check correctness, oh & ow or h & w?
                        auto oOffset = ww + ow * (hh + oh * (ff + f * nn));
                        optr[oOffset] = val;
                    }
            }
        }
    }
};

REGISTER_KERNEL(Device::CPU, OpType::Conv, DataType::UInt32,
                NaiveConv<uint32_t>, "ConvNaive_CPU_uint32");
REGISTER_KERNEL(Device::CPU, OpType::Conv, DataType::Float32, NaiveConv<float>,
                "ConvNaive_CPU_float32");

} // namespace infini
