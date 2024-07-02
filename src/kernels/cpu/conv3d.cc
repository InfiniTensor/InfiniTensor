#include "core/kernel.h"
#include "operators/conv.h"

namespace infini {

class NaiveConv3d : public CpuKernelWithoutConfig {
    template <typename T>
    void doCompute(const Operator &_op, const RuntimeObj *context) const {
        auto op = as<Conv3dObj>(_op);
        T *iptr = op->getInputs(0)->getRawDataPtr<T *>();
        T *wptr = op->getInputs(1)->getRawDataPtr<T *>();
        T *optr = op->getOutput()->getRawDataPtr<T *>();
        //  Clang will give an error of " reference to local binding 'sh'
        //  declared in enclosing function" if we write like this:
        //        auto [n, c, d, h, w, f, q, r, s] = op->getNCDHWFRSQ();
        int n, c, d, h, w, f, q, r, s;
        std::tie(n, c, d, h, w, f, q, r, s) = op->getNCDHWFQRS();
        int pd, ph, pw, sd, sh, sw, dd_, dh, dw;
        std::tie(pd, ph, pw, sd, sh, sw, dd_, dh, dw) =
            op->getPadStrideDilation();
        int cpg = op->getChannelPerGroup();
        int g = op->getNumGroups();
        IT_ASSERT(f % g == 0, "Illegal number of channel");
        auto outDim = op->getOutput()->getDims();
        int od = outDim[2], oh = outDim[3], ow = outDim[4];
        for (int nn = 0; nn < n; nn++) {
#pragma omp parallel for
            for (int ff = 0; ff < f; ff++) {
                for (int dd = 0; dd < od; dd++) {
                    for (int hh = 0; hh < oh; hh++) {
                        for (int ww = 0; ww < ow; ww++) {
                            int gidx = ff / (f / g);
                            T val = 0;
                            for (int cc = 0; cc < cpg; cc++) {
                                for (int qq = 0; qq < q; qq++) {
                                    for (int rr = 0; rr < r; rr++) {
                                        for (int ss = 0; ss < s; ss++) {
                                            // clang-format off
                                            int posD = dd * sd + qq * dd_ - pd;
                                            int posH = hh * sh + rr * dh - ph;
                                            int posW = ww * sw + ss * dw - pw;
                                            if (posD < 0 || posD >= d ||
                                                posH < 0 || posH >= h ||
                                                posW < 0 || posW >= w)
                                                continue;
                                            auto iOffset = posW + w * (posH + h * (posD + d * ((cc + gidx * cpg) + c * nn)));
                                            auto wOffset = ss + s * (rr + r * (qq + q * (cc + cpg * ff)));
                                            auto inputVal = iptr[iOffset];
                                            auto weightVal = wptr[wOffset];
                                            val += weightVal * inputVal;
                                            // clang-format on
                                        }
                                    }
                                }
                            }
                            // TODO: Check correctness:
                            // od & oh & ow or d & h & w?
                            auto oOffset =
                                ww + ow * (hh + oh * (dd + od * (ff + f * nn)));
                            optr[oOffset] = val;
                        }
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

REGISTER_KERNEL(Device::CPU, OpType::Conv3d, NaiveConv3d, "Conv3dNaive_CPU");

} // namespace infini
