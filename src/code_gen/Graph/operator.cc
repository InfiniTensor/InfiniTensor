#include "code_gen/operator.h"
#include "code_gen/common.h"
#include "code_gen/custom_ops.h"
#include "code_gen/graph.h"
#include "code_gen/perf_engine.h"
#include "code_gen/tensor.h"
#include <chrono>
#include <cstdlib>

namespace tpm {

namespace ch = std::chrono;

ConvOp::ConvOp(Tensor *input, Tensor *weight, Tensor *output, int ph, int pw,
               int sh, int sw, int dh, int dw, Tensor *bias, ActType act)
    : Operator(Conv, {input, weight}, {output}), ph(ph), pw(pw), sh(sh), sw(sw),
      dh(dh), dw(dw), bias(bias), act(act), padding(Other) {
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    computeShape();
    setPaddingMode();
    initHash();
    assert(output->getDims().size() == 4);
}

ConvOp::ConvOp(Tensor *input, Tensor *weight, int ph, int pw, int sh, int sw,
               int dh, int dw, Tensor *bias, ActType act)
    : Operator(Conv, {input, weight}, {}), ph(ph), pw(pw), sh(sh), sw(sw),
      dh(dh), dw(dw), bias(bias), act(act), padding(Other) {
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    outputs.emplace_back(new Tensor());
    computeShape();
    setPaddingMode();
    initHash();
}

ConvOp::ConvOp(int ph, int pw, int sh, int sw, int dh, int dw, Tensor *bias,
               ActType act)
    : Operator(Conv), ph(ph), pw(pw), sh(sh), sw(sw), dh(dh), dw(dw),
      bias(bias), act(act), padding(Other) {
    initHash();
}

ConvOp::ConvOp(Tensor *input, Tensor *weight, Tensor *output, PaddingMode pm,
               int sh, int sw, int dh, int dw, Tensor *bias, ActType act)
    : Operator(Conv, {input, weight}, {output}), sh(sh), sw(sw), dh(dh), dw(dw),
      bias(bias), act(act), padding(pm) {
    assert(pm != Other);
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    assert(output->getDims().size() == 4);
    // set padding size
    computeShape();
    initHash();
}
ConvOp::ConvOp(Tensor *input, Tensor *weight, PaddingMode pm, int sh, int sw,
               int dh, int dw, Tensor *bias, ActType act)
    : Operator(Conv, {input, weight}, {}), sh(sh), sw(sw), dh(dh), dw(dw),
      bias(bias), act(act), padding(pm) {
    assert(pm != Other);
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}
ConvOp::ConvOp(PaddingMode pm, int sh, int sw, int dh, int dw, Tensor *bias,
               ActType act)
    : Operator(Conv), sh(sh), sw(sw), dh(dh), dw(dw), bias(bias), act(act),
      padding(pm) {
    // assert(pm != Other);
    initHash();
}

ConvOp::ConvOp(const ConvOp &rhs)
    : Operator(rhs), ph(rhs.ph), pw(rhs.pw), sh(rhs.sh), sw(rhs.sw), dh(rhs.dh),
      dw(rhs.dw), bias(rhs.bias), act(rhs.act), padding(rhs.padding) {}

void ConvOp::initHash() {
    hash = type;
    hash = hashAppend(hash, padding);
    if (padding == Other) {
        hash = hashAppend(hash, ph);
        hash = hashAppend(hash, pw);
    }
    hash = hashAppend(hash, sh);
    hash = hashAppend(hash, sw);
    hash = hashAppend(hash, dh);
    hash = hashAppend(hash, dw);
    hash = hashPack(hash);
}

Tensor *ConvOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto input = inputs[0], weight = inputs[1], output = outputs[0];
    auto n = input->getDims()[0];
    auto c = input->getDims()[1];
    auto h = input->getDims()[2];
    auto w = input->getDims()[3];
    auto f = weight->getDims()[0];
    auto cpg = weight->getDims()[1];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    auto g = c / cpg;
    if (f % g != 0)
        return nullptr;
    output->dataMalloc();
    auto outDim = output->getDims();
    auto oh = outDim[2], ow = outDim[3];
    auto iptr = input->getDataPtr(), wptr = weight->getDataPtr(),
         optr = output->getDataPtr();
    for (int nn = 0; nn < n; nn++) {
#pragma omp parallel for
        for (int ff = 0; ff < f; ff++) {
            for (int hh = 0; hh < oh; hh++)
                for (int ww = 0; ww < ow; ww++) {
                    int gidx = ff / (f / g);
                    VType val = 0;
                    for (int cc = 0; cc < cpg; cc++)
                        for (int rr = 0; rr < r; rr++)
                            for (int ss = 0; ss < s; ss++) {
                                int posH = hh * sh + rr * dh - ph;
                                int posW = ww * sw + ss * dw - pw;
                                // VType weightVal =
                                //     weight->getData({ff, cc, rr, ss});
                                // VType inputVal = input->getData(
                                //     {nn, cc + gidx * cpg, posH, posW});
                                if (posH < 0 || posH >= h || posW < 0 ||
                                    posW >= w)
                                    continue;
                                auto iOffset =
                                         posW +
                                         w * (posH +
                                              h * ((cc + gidx * cpg) + c * nn)),
                                     wOffset =
                                         ss + s * (rr + r * (cc + cpg * ff));
                                // auto iOffset = input->getOffset(
                                //          {nn, cc + gidx * cpg, posH, posW}),
                                //      wOffset =
                                //          weight->getOffset({ff, cc, rr, ss});
                                auto inputVal = iptr[iOffset],
                                     weightVal = wptr[wOffset];
                                val += weightVal * inputVal;
                            }
                    // output->setData({nn, ff, hh, ww}, val);
                    // auto oOffset = output->getOffset({nn, ff, hh, ww});
                    // TODO: check correctness, oh & ow or h & w?
                    auto oOffset = ww + ow * (hh + oh * (ff + f * nn));
                    optr[oOffset] = val;
                }
        }
    }
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
ConvOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty(), DimRange::getEmpty()},
                []() { return true; }};
    auto input = inputs[0], weight = inputs[1];
    auto c = input->getDims()[1];
    auto f = weight->getDims()[0];
    auto cpg = weight->getDims()[1];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    auto g = c / cpg;
    if (f % g != 0)
        return {};
    auto outDim = outputs[0]->getDims();
    // TODO: call gpu compute
    if (!dr.isSinglePos()) {
        return {{DimRange::getAllPos(), DimRange::getAllPos()},
                [this]() { return compute() != nullptr; }};
    } else {
        if (dr.getBegin().size() != 4 /*|| dr.getEnd().size() != 4*/)
            return {};
        return {
            {DimRange::getAllPos(), DimRange::getAllPos()},
            [this, f, g, cpg, r, s, dr]() {
                auto &pos = dr.getBegin();
                auto nn = pos[0], ff = pos[1], hh = pos[2], ww = pos[3];
                auto input = inputs[0], weight = inputs[1], output = outputs[0];
                int gidx = ff / (f / g);
                VType val = 0;
                for (int cc = 0; cc < cpg; cc++)
                    for (int rr = 0; rr < r; rr++)
                        for (int ss = 0; ss < s; ss++) {
                            int posH = hh * sh + rr * dh - ph;
                            int posW = ww * sw + ss * dw - pw;
                            VType weightVal = weight->getData({ff, cc, rr, ss});
                            VType inputVal = input->getData(
                                {nn, cc + gidx * cpg, posH, posW});
                            val += weightVal * inputVal;
                        }
                output->dataMalloc();
                return output->setData({nn, ff, hh, ww}, val);
            }};
    }
}

Dim ConvOp::computeShape() {
    auto input = inputs[0], weight = inputs[1], output = outputs[0];
    auto n = input->getDims()[0];
    auto h = input->getDims()[2];
    auto w = input->getDims()[3];
    auto f = weight->getDims()[0];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // Set padding size
    if (padding == Other) {
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    } else if (padding == Same) {
        oh = h / sh;
        ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (padding == Valid) {
        ph = 0;
        pw = 0;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    }
    auto ret = {on, oc, oh, ow};
    output->setDims(ret);
    output->setType(Tensor::Input);
    return ret;
}

Dim ConvOp::computeOutputPenalty(const Dim &p) {
    assert(p.size() == 4);
    auto np = p[0], hp = p[2], wp = p[3];
    auto input = inputs[0], weight = inputs[1], output = outputs[0];
    auto n = input->getDims()[0] + np;
    auto h = input->getDims()[2] + hp;
    auto w = input->getDims()[3] + wp;
    auto f = weight->getDims()[0];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    // Set padding size
    if (padding == Other) {
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    } else if (padding == Same) {
        oh = h / sh;
        ow = w / sw;
        ph = (h - oh * sh + (r - sh) * dh) / 2;
        pw = (w - ow * sw + (s - sw) * dw) / 2;
    } else if (padding == Valid) {
        ph = 0;
        pw = 0;
        oh = (h - (r - sh) * dh + ph * 2) / sh;
        ow = (w - (s - sw) * dw + pw * 2) / sw;
    }
    auto outDim = output->getDims();
    return {on - outDim[0], oc - outDim[1], oh - outDim[2], ow - outDim[3]};
}

// Only called by constructors which explicitly set padding size
// computeShape() is called in constructor
void ConvOp::setPaddingMode() {
    auto iDim = inputs[0]->getDims();
    auto oDim = outputs[0]->getDims();
    if (iDim[2] == oDim[2] && iDim[3] == oDim[3])
        padding = Same;
    else if (ph == 0 && pw == 0)
        padding = Valid;
}

// void ConvOp::setPaddingSize() {}

bool ConvOp::checkValid(const TensorVec &inputs) {
    auto input = inputs[0], weight = inputs[1];
    assert(input != nullptr && weight != nullptr);
    if (input->getType() != Tensor::Input ||
        weight->getType() != Tensor::Weight)
        return false;
    if (input->getDims().size() != 4 || weight->getDims().size() != 4)
        return false;
    if (input->getDims()[1] % weight->getDims()[1] != 0)
        return false;
    return true;
}

double ConvOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    constexpr int N_ALGO = 8;
    constexpr cudnnConvolutionFwdAlgo_t ALGOS[N_ALGO] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED};

    auto input = inputs[0], weight = inputs[1];
    auto n = input->getDims()[0] + pe->withPenalty() * input->getPenalty()[0];
    auto c = input->getDims()[1] + pe->withPenalty() * input->getPenalty()[1];
    auto h = input->getDims()[2] + pe->withPenalty() * input->getPenalty()[2];
    auto w = input->getDims()[3] + pe->withPenalty() * input->getPenalty()[3];
    auto f = weight->getDims()[0];
    auto cpg = weight->getDims()[1];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    auto g = c / cpg;

    ConvArgs args = getArgs(pe->withPenalty());

    if (pe->checkOpPerf(Conv, args)) {
        return pe->getOpPerf(Conv, args);
    }

    int channelsPerGrp = cpg, channels = c;

    // get inputs
    cudnnTensorDescriptor_t inDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, channels, h, w));

    float *inData;
    inData = pe->getInputPtr();

    // get kernels
    cudnnFilterDescriptor_t knDesc;
    checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
    checkCudnnError(cudnnSetFilter4dDescriptor(
        knDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, f, channelsPerGrp, r, s));

    float *knData;
    knData = pe->getWeightPtr();

    // get bias
    cudnnTensorDescriptor_t biasDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT, 1, f, 1, 1));

    float *biasData;
    biasData = pe->getBiasPtr();

    // get convlution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCudnnError(cudnnSetConvolution2dDescriptor(
        convDesc, ph, pw, sh, sw, dh, dw, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
    if (g > 1) {
        checkCudnnError(cudnnSetConvolutionGroupCount(convDesc, g));
    }

    // get activation descriptor
    cudnnActivationDescriptor_t actDesc;
    checkCudnnError(cudnnCreateActivationDescriptor(&actDesc));
    // NOT_PROPAGATE_NAN is requierd by cudnnConvolotionBiasActivationForward
    switch (act) {
    case Relu:
        checkCudnnError(cudnnSetActivationDescriptor(
            actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
        break;
    case Sigmoid:
        checkCudnnError(cudnnSetActivationDescriptor(
            actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
        break;
    case None:
        checkCudnnError(cudnnSetActivationDescriptor(
            actDesc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0));
        break;
    default:
        assert(false);
    }

    // get outputs
    int outn, outc, outh, outw;
    checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
        convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));
    cudnnTensorDescriptor_t outDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outn, outc, outh, outw));

    float *outData;
    outData = pe->getOutputPtr();

    ConvResult best;
    best.time = INFINITY;
    for (int i = 0; i < N_ALGO; i++) {
        // get workspace
        size_t wsSize;
        auto stat = cudnnGetConvolutionForwardWorkspaceSize(
            pe->cudnnHandle(), inDesc, knDesc, convDesc, outDesc, ALGOS[i],
            &wsSize);
        if (stat != CUDNN_STATUS_SUCCESS) {
            continue;
        }
        // assert(wsSize < (size_t)3 * 1024 * 1024 * 1024);
        if (wsSize >= (size_t)10 * 1024 * 1024 * 1024)
            continue;
        float *wsData;
        wsData = pe->getWorkspace();

        // perform convolution
        double durtime = 0.0, durtime_fuse = 0.0;
        float alpha = 1.f, beta = 0.f;
        ch::time_point<ch::high_resolution_clock, ch::nanoseconds> beg, end;
        beg = ch::high_resolution_clock::now();
        for (int j = 0; j < rounds + warmupRounds; ++j) {
            cudnnStatus_t stat;
            // w/o bias & act
            if (j == warmupRounds) {
                checkCudaError(cudaDeviceSynchronize());
                beg = ch::high_resolution_clock::now();
            }
            stat = cudnnConvolutionForward(
                pe->cudnnHandle(), &alpha, inDesc, inData, knDesc, knData,
                convDesc, ALGOS[i], wsData, wsSize, &beta, outDesc, outData);
            if (stat != CUDNN_STATUS_SUCCESS) {
                // checkCudnnError(stat);
                // Do not checkCudnnError since not all algorithms are supported
                durtime = INFINITY;
                break;
            }

            // // bias
            // if (bias != nullptr) {
            //     auto sz = outputs[0]->size();
            //     // TODO: element wise
            //     t += sz * 2 / 400;
            // }

            // // act
            // if (act != None) {
            //     checkCudaError(cudaDeviceSynchronize());
            //     beg = ch::high_resolution_clock::now();
            //     stat = cudnnActivationForward(pe->cudnnHandle(), actDesc,
            //                                   &alpha, inDesc, inData, &beta,
            //                                   outDesc, outData);
            //     checkCudaError(cudaDeviceSynchronize());
            //     end = ch::high_resolution_clock::now();
            //     if (stat != CUDNN_STATUS_SUCCESS) {
            //         durtime = INFINITY;
            //         break;
            //     }
            //     t +=
            //         ch::duration_cast<ch::duration<double>>(end -
            //         beg).count() * 1000; // ms
            // }
        }
        checkCudaError(cudaDeviceSynchronize());
        end = ch::high_resolution_clock::now();
        if (durtime == 0) {
            auto t =
                ch::duration_cast<ch::duration<double>>(end - beg).count() *
                1000; // ms
            durtime = double(t) / rounds;
        }
        if (durtime < best.time)
            best = ConvResult{durtime, ALGOS[i], wsSize, false};

        // w/ bias & act
        for (int j = 0; j < rounds + warmupRounds; ++j) {
            cudnnStatus_t stat;
            if (j == warmupRounds) {
                checkCudaError(cudaDeviceSynchronize());
                beg = ch::high_resolution_clock::now();
            }
            stat = cudnnConvolutionBiasActivationForward(
                pe->cudnnHandle(), &alpha, inDesc, inData, knDesc, knData,
                convDesc, ALGOS[i], wsData, wsSize, &beta, outDesc, outData,
                biasDesc, biasData, actDesc, outDesc, outData);
            if (stat != CUDNN_STATUS_SUCCESS) {
                // checkCudnnError(stat);
                // Do not checkCudnnError since not all algorithms are supported
                durtime_fuse = INFINITY;
                break;
            }
        }
        checkCudaError(cudaDeviceSynchronize());
        end = ch::high_resolution_clock::now();
        if (durtime_fuse == 0) {
            auto t =
                ch::duration_cast<ch::duration<double>>(end - beg).count() *
                1000; // ms
            durtime_fuse = double(t) / rounds;
        }
        if (durtime_fuse < best.time) {
            best = ConvResult{durtime_fuse, ALGOS[i], wsSize, true};
        }
        // std::cout << "[perf conv] " << ALGOS[i] << ", "
        //           << "act: " << act << ", unfused, " << durtime << ", fused,
        //           "
        //           << durtime_fuse << std::endl;
    }

    // finalize
    checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
    checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
    checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
    checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));

    pe->saveOpPerf(Conv, args, best);

    return best.time;
}

bool ConvOp::same(const ConvOp &rhs) {
    bool ret = true;
    if (padding == Other)
        ret &= ph == rhs.ph && pw == rhs.pw;
    else
        ret &= padding == rhs.padding;
    ret &= sh == rhs.sh && sw == rhs.sw;
    ret &= dh == rhs.dh && dw == rhs.dw;
    return ret;
}

std::string ConvOp::toString() const {
    std::ostringstream os;
    os << "Conv[" << hash << "]";
    os << "(";
    if (inputs.size() == 2) {
        os << dimToString(inputs[0]->getDims()) << ",";
        os << dimToString(inputs[1]->getDims()) << ",";
    }
    os << dimToString(inputs[0]->getPenalty()) << ",";
    os << dimToString(outputs[0]->getPenalty()) << ",";

    bool flag =
        (inputs[0]->getPenalty().size() != outputs[0]->getPenalty().size());
    if (!flag) {
        for (int i = 0; i < int(inputs[0]->getPenalty().size()); i++) {
            if (inputs[0]->getPenalty()[i] != outputs[0]->getPenalty()[i]) {
                flag = 1;
            }
        }
    }
    if (flag) {
        os << "[INVALID_PENALTY]";
    }
    os << "p=[" << ph << "," << pw << "],";
    os << "s=[" << sh << "," << sw << "],";
    os << "d=[" << dh << "," << dw << "],";
    os << "act=" << act << ",";
    os << "input=" << inputs[0]->getHash() << ",";
    os << "weight=" << inputs[1]->getHash() << ",";
    os << "output=" << outputs[0]->getHash() << ")";
    return os.str();
}

void ConvOp::inferSplittingPoints() {
    auto inputSplittingPoints = getInputs()[0]->getSplittingPoints();
    // : Operator(Conv, {input, weight}, {output}), ph(ph), pw(pw), sh(sh),
    // sw(sw),
    //   dh(dh), dw(dw), bias(bias), act(act) {
    assert(inputSplittingPoints->size() == 4);
    for (auto &points : *inputSplittingPoints)
        assert(points.empty());

    SplittingPoints splittingPoints(4);
    int h = inputs[0]->getDims()[2], w = inputs[0]->getDims()[3];
    int kh = inputs[1]->getDims()[2], kw = inputs[1]->getDims()[3];

    auto insert = [](std::vector<int> &points, int h, int kh, int ph, int sh,
                     int dh) {
        int last_left_point = (ph + sh - 1) / sh;
        // int first_right_point = (h - 1 - dh * (kh - 1)) / sh + 1;
        int first_right_point = (ph + h - dh * (kh - 1) + sh - 1) / sh;
        for (int i = 1; i <= last_left_point; ++i)
            points.emplace_back(i);
        // omit the boudary point
        for (int i = first_right_point; i < h; ++i)
            points.emplace_back(i);
    };
    insert(splittingPoints[2], h, kh, ph, sh, dh);
    insert(splittingPoints[3], w, kw, pw, sw, dw);
    getOutput()->setSplittingPoints(std::move(splittingPoints));
}

MatmulOp::MatmulOp(Tensor *A, Tensor *B, bool transA, bool transB, Tensor *bias,
                   ActType act)
    : Operator(Matmul, {A, B}, {}), transA(transA), transB(transB), bias(bias),
      act(act) {
    checkAndSetTensorTypeForConstructor(A, B);
    assert(checkValid({A, B}));
    outputs.emplace_back(new Tensor);
    computeShape();
    initHash();
}

MatmulOp::MatmulOp(bool transA, bool transB, Tensor *bias, ActType act)
    : Operator(Matmul), transA(transA), transB(transB), bias(bias), act(act) {
    initHash();
}

void MatmulOp::initHash() {
    hash = type;
    hash = hashAppend(hash, transA);
    hash = hashAppend(hash, transB);
    hash = hashPack(hash);
}

// This function will only be called by constructor to set the tensor type
// TODO: this function can be removed when more tensor types are introduced
void MatmulOp::checkAndSetTensorTypeForConstructor(Tensor *A, Tensor *B) {
    if (A->getType() == Tensor::Input && B->getType() == Tensor::Input)
        B->setType(Tensor::Weight);
}

Tensor *MatmulOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto A = inputs[0], B = inputs[1], C = outputs[0];
    auto b = A->getDims()[0];
    auto m = transA ? A->getDims()[2] : A->getDims()[1];
    auto n = transB ? B->getDims()[1] : B->getDims()[2];
    auto k = transA ? A->getDims()[1] : A->getDims()[2];
    C->dataMalloc();
    auto Aptr = A->getDataPtr(), Bptr = B->getDataPtr(), Cptr = C->getDataPtr();
#pragma omp parallel for collapse(2)
    for (int bb = 0; bb < b; ++bb) {
        for (int mm = 0; mm < m; ++mm) {
            for (int nn = 0; nn < n; ++nn) {
                VType tmp = 0;
                for (int kk = 0; kk < k; ++kk) {
                    // auto lhs = transA ? A->getData({bb, kk, mm})
                    //                   : A->getData({bb, mm, kk});
                    auto lhs = transA ? Aptr[mm + m * (kk + k * bb)]
                                      : Aptr[kk + k * (mm + m * bb)];
                    // auto rhs = transB ? B->getData({bb, nn, kk})
                    //                   : B->getData({bb, kk, nn});
                    auto rhs = transB ? Bptr[kk + k * (nn + n * bb)]
                                      : Bptr[nn + n * (kk + k * bb)];
                    tmp += lhs * rhs;
                }
                // C->setData({bb, mm, nn}, tmp);
                Cptr[nn + n * (mm + m * bb)] = tmp;
            }
        }
    }
    C->setComputed();
    return C;
}

void MatmulOp::dimExtend(Tensor *t) {
    auto dm = t->getDims();
    if (dm.size() == 2) {
        dm.insert(dm.begin(), 1);
        t->setDims(dm);
    }
}

std::pair<std::vector<DimRange>, std::function<bool()>>
MatmulOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty(), DimRange::getEmpty()},
                []() { return true; }};
    if (!dr.isSinglePos()) {
        return {{DimRange::getAllPos(), DimRange::getAllPos()},
                [this]() { return compute() != nullptr; }};
    } else {
        auto pos = dr.getBegin();
        if (pos.size() != outputs[0]->getDims().size())
            return {};
        auto bb = pos[0], mm = pos[1], nn = pos[2];
        auto k = transA ? inputs[0]->getDims()[1] : inputs[0]->getDims()[2];
        return {{transA ? DimRange({bb, 0, mm}, {bb, k - 1, mm})
                        : DimRange({bb, mm, 0}, {bb, mm, k - 1}),
                 transB ? DimRange({bb, nn, 0}, {bb, nn, k - 1})
                        : DimRange({bb, 0, nn}, {bb, k - 1, nn})},
                [this, bb, mm, nn, k]() {
                    VType val = 0;
                    auto A = inputs[0], B = inputs[1], C = outputs[0];
                    for (int kk = 0; kk < k; ++kk) {
                        auto lhs = transA ? A->getData({bb, kk, mm})
                                          : A->getData({bb, mm, kk});
                        auto rhs = transB ? B->getData({bb, nn, kk})
                                          : B->getData({bb, kk, nn});
                        val += lhs * rhs;
                    }
                    C->dataMalloc();
                    return C->setData({bb, mm, nn}, val);
                }};
    }
}

bool MatmulOp::checkValid(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    if (A->getType() == Tensor::Weight && B->getType() == Tensor::Weight)
        return false;
    if (A->getDims().size() != 3 || B->getDims().size() != 3) {
        return false;
    }
    if (A->getDims()[0] != B->getDims()[0]) {
        return false;
    }
    if ((transA ? A->getDims()[1] : A->getDims()[2]) !=
        (transB ? B->getDims()[2] : B->getDims()[1])) {
        return false;
    }
    return true;
}

Dim MatmulOp::computeShape() {
    auto A = inputs[0], B = inputs[1], C = outputs[0];
    auto b = A->getDims()[0];
    auto m = transA ? A->getDims()[2] : A->getDims()[1];
    auto n = transB ? B->getDims()[1] : B->getDims()[2];
    auto ret = {b, m, n};
    C->setDims(ret);
    C->setType(Tensor::Input);
    return ret;
}

double MatmulOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    constexpr int N_ALGO = 24;
    constexpr cublasGemmAlgo_t ALGOS[N_ALGO] = {
        CUBLAS_GEMM_ALGO0,  CUBLAS_GEMM_ALGO1,  CUBLAS_GEMM_ALGO2,
        CUBLAS_GEMM_ALGO3,  CUBLAS_GEMM_ALGO4,  CUBLAS_GEMM_ALGO5,
        CUBLAS_GEMM_ALGO6,  CUBLAS_GEMM_ALGO7,  CUBLAS_GEMM_ALGO8,
        CUBLAS_GEMM_ALGO9,  CUBLAS_GEMM_ALGO10, CUBLAS_GEMM_ALGO11,
        CUBLAS_GEMM_ALGO12, CUBLAS_GEMM_ALGO13, CUBLAS_GEMM_ALGO14,
        CUBLAS_GEMM_ALGO15, CUBLAS_GEMM_ALGO16, CUBLAS_GEMM_ALGO17,
        CUBLAS_GEMM_ALGO18, CUBLAS_GEMM_ALGO19, CUBLAS_GEMM_ALGO20,
        CUBLAS_GEMM_ALGO21, CUBLAS_GEMM_ALGO22, CUBLAS_GEMM_ALGO23,
    };

    auto A = inputs[0], B = inputs[1];
    auto b = A->getDims()[0];
    auto m = transA ? A->getDims()[2] : A->getDims()[1];
    auto n = transB ? B->getDims()[1] : B->getDims()[2];
    auto k = transA ? A->getDims()[1] : A->getDims()[2];

    MatmulArgs args = getArgs();

    if (pe->checkOpPerf(Matmul, args)) {
        return pe->getOpPerf(Matmul, args);
    }

    // cublas uses column major, we are computing C^T
    // C^T = B^T A^T
    // In the following notation, `a` means our actual matrix A, i.e. B for
    // cublas
    float *dA, *dB, *dC;
    dA = pe->getMatA();
    dB = pe->getMatB();
    dC = pe->getMatC();
    auto opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N; // BLAS_N = col major
    auto opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = transA ? m : k, ldb = transB ? k : n, ldc = n;
    const float alpha = 1.f, beta = 0.f;

    MatmulResult best;
    best.time = INFINITY;

    ch::time_point<ch::high_resolution_clock, ch::nanoseconds> beg, end;
    for (int i = -1; i < N_ALGO; i++) {
        double durtime = 0.0;
        for (int j = 0; j < rounds + warmupRounds; ++j) {
            if (j == warmupRounds) {
                checkCudaError(cudaDeviceSynchronize());
                beg = ch::high_resolution_clock::now();
            }
            auto stat = cublasGemmStridedBatchedEx(
                pe->cublasHandle(), opB, opA, n, m, k, &alpha, dB, CUDA_R_32F,
                ldb, k * n, dA, CUDA_R_32F, lda, m * k, &beta, dC, CUDA_R_32F,
                ldc, m * n, b, CUDA_R_32F, ALGOS[i]);
            if (stat != CUBLAS_STATUS_SUCCESS) {
                durtime = INFINITY;
                break;
            }
        }
        checkCudaError(cudaDeviceSynchronize());
        end = ch::high_resolution_clock::now();
        if (durtime == 0) {
            durtime =
                ch::duration_cast<ch::duration<double>>(end - beg).count() *
                1000; // ms
        }
        durtime /= rounds;
        if (durtime < best.time) {
            best = MatmulResult{durtime, true, ALGOS[i]};
        }
    }

    if (b == 1) {
        for (int i = 0; i < N_ALGO; i++) {
            double durtime = 0.0;
            for (int j = 0; j < rounds + warmupRounds; ++j) {
                if (j == warmupRounds) {
                    checkCudaError(cudaDeviceSynchronize());
                    beg = ch::high_resolution_clock::now();
                }
                auto stat = cublasGemmEx(pe->cublasHandle(), opB, opA, n, m, k,
                                         &alpha, dB, CUDA_R_32F, ldb, dA,
                                         CUDA_R_32F, lda, &beta, dC, CUDA_R_32F,
                                         ldc, CUDA_R_32F, ALGOS[i]);
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    durtime = INFINITY;
                    break;
                }
            }
            checkCudaError(cudaDeviceSynchronize());
            end = ch::high_resolution_clock::now();
            if (durtime == 0) {
                durtime =
                    ch::duration_cast<ch::duration<double>>(end - beg).count() *
                    1000; // ms
            }
            durtime /= rounds;
            if (durtime < best.time) {
                best = MatmulResult{durtime, false, ALGOS[i]};
            }
        }
    }

    pe->saveOpPerf(Matmul, args, best);
    return best.time;
}

void MatmulOp::inferSplittingPoints() {
    // Assume no prior splitting points
    for (auto tensor : inputs) {
        for (auto v : *tensor->getSplittingPoints())
            assert(v.size() == 0);
    }
    outputs[0]->initSplittingPoints();
}

ConvTransOp::ConvTransOp(Tensor *input, Tensor *weight, Tensor *output, int ph,
                         int pw, int sh, int sw, int dh, int dw, int oph,
                         int opw, Tensor *bias, ActType act)
    : Operator(ConvTrans, {input, weight}, {output}), ph(ph), pw(pw), sh(sh),
      sw(sw), dh(dh), dw(dw), oph(oph), opw(opw), bias(bias), act(act),
      padding(Other) {
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    computeShape();
    setPaddingMode();
    initHash();
    assert(output->getDims().size() == 4);
}

ConvTransOp::ConvTransOp(Tensor *input, Tensor *weight, int ph, int pw, int sh,
                         int sw, int dh, int dw, int oph, int opw, Tensor *bias,
                         ActType act)
    : Operator(ConvTrans, {input, weight}, {}), ph(ph), pw(pw), sh(sh), sw(sw),
      dh(dh), dw(dw), oph(oph), opw(opw), bias(bias), act(act), padding(Other) {
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    outputs.emplace_back(new Tensor());
    computeShape();
    setPaddingMode();
    initHash();
}

ConvTransOp::ConvTransOp(int ph, int pw, int sh, int sw, int dh, int dw,
                         int oph, int opw, Tensor *bias, ActType act)
    : Operator(ConvTrans), ph(ph), pw(pw), sh(sh), sw(sw), dh(dh), dw(dw),
      oph(oph), opw(opw), bias(bias), act(act), padding(Other) {
    initHash();
}

ConvTransOp::ConvTransOp(Tensor *input, Tensor *weight, Tensor *output,
                         PaddingMode pm, int sh, int sw, int dh, int dw,
                         int oph, int opw, Tensor *bias, ActType act)
    : Operator(ConvTrans, {input, weight}, {output}), sh(sh), sw(sw), dh(dh),
      dw(dw), oph(oph), opw(opw), bias(bias), act(act), padding(pm) {
    assert(pm != Other);
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    assert(output->getDims().size() == 4);
    // set padding size
    computeShape();
    initHash();
}
ConvTransOp::ConvTransOp(Tensor *input, Tensor *weight, PaddingMode pm, int sh,
                         int sw, int dh, int dw, int oph, int opw, Tensor *bias,
                         ActType act)
    : Operator(ConvTrans, {input, weight}, {}), sh(sh), sw(sw), dh(dh), dw(dw),
      oph(oph), opw(opw), bias(bias), act(act), padding(pm) {
    assert(pm != Other);
    weight->setType(Tensor::Weight);
    assert(checkValid({input, weight}));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}
ConvTransOp::ConvTransOp(PaddingMode pm, int sh, int sw, int dh, int dw,
                         int oph, int opw, Tensor *bias, ActType act)
    : Operator(ConvTrans), sh(sh), sw(sw), dh(dh), dw(dw), oph(oph), opw(opw),
      bias(bias), act(act), padding(pm) {
    // assert(pm != Other);
    initHash();
}

ConvTransOp::ConvTransOp(const ConvTransOp &rhs)
    : Operator(rhs), ph(rhs.ph), pw(rhs.pw), sh(rhs.sh), sw(rhs.sw), dh(rhs.dh),
      dw(rhs.dw), oph(rhs.oph), opw(rhs.opw), bias(rhs.bias), act(rhs.act),
      padding(rhs.padding) {}

void ConvTransOp::initHash() {
    hash = type;
    hash = hashAppend(hash, padding);
    if (padding == Other) {
        hash = hashAppend(hash, ph);
        hash = hashAppend(hash, pw);
    }
    hash = hashAppend(hash, sh);
    hash = hashAppend(hash, sw);
    hash = hashAppend(hash, dh);
    hash = hashAppend(hash, dw);
    hash = hashAppend(hash, oph);
    hash = hashAppend(hash, opw);
    hash = hashPack(hash);
}

// input {n, h, w, f} weight {r, s, f, c} output {n, h, w, c}
Tensor *ConvTransOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto input = inputs[0], weight = inputs[1], output = outputs[0];
    auto n = input->getDims()[0];
    auto h = input->getDims()[1];
    auto w = input->getDims()[2];
    auto f = input->getDims()[3];

    auto r = weight->getDims()[0];
    auto s = weight->getDims()[1];
    auto cpg = weight->getDims()[3];

    output->dataMalloc();
    auto outDim = output->getDims();
    auto oh = outDim[2], ow = outDim[3];
    for (int nn = 0; nn < n; nn++) {
#pragma omp parallel for
        for (int cc = 0; cc < cpg; cc++) {
            for (int hh = 0; hh < oh; hh++)
                for (int ww = 0; ww < ow; ww++) {
                    VType val = 0;
                    for (int ff = 0; ff < f; ff++)
                        for (int rr = 0; rr < r; rr++)
                            for (int ss = 0; ss < s; ss++) {
                                int ah = ((h + 2 * ph - r) % sh);
                                int aw = ((w + 2 * pw - s) % sw);
                                if (r - ah < rr || s - aw < ss)
                                    continue;
                                int posH = hh + rr * dh - (r - ph - 1);
                                int posW = ww + ss * dw - (s - pw - 1);
                                if (posH % sh == 0 && posW % sw == 0) {
                                    posH /= sh;
                                    posW /= sw;
                                    if (posH < 0 || posH >= h || posW < 0 ||
                                        posW >= w)
                                        continue;
                                    auto inputVal =
                                        input->getData({nn, posH, posW, ff});
                                    auto weightVal = weight->getData(
                                        {(r - 1 - rr), (s - 1 - ss), ff, cc});
                                    val += weightVal * inputVal;
                                }
                            }
                    output->setData({nn, hh, ww, cc}, val);
                }
        }
    }
    output->setComputed();
    return output;
}

// input {n, h, w, f} weight {r, s, f, c} output {n, h, w, c}
std::pair<std::vector<DimRange>, std::function<bool()>>
ConvTransOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty(), DimRange::getEmpty()},
                []() { return true; }};
    auto input = inputs[0], weight = inputs[1];
    auto h = input->getDims()[1];
    auto w = input->getDims()[2];
    auto f = input->getDims()[3];
    auto r = weight->getDims()[0];
    auto s = weight->getDims()[1];
    auto cpg = weight->getDims()[3];
    auto outDim = outputs[0]->getDims();
    // TODO: call gpu compute
    if (!dr.isSinglePos()) {
        return {{DimRange::getAllPos(), DimRange::getAllPos()},
                [this]() { return compute() != nullptr; }};
    } else {
        if (dr.getBegin().size() != 4 /*|| dr.getEnd().size() != 4*/)
            return {};
        return {{DimRange::getAllPos(), DimRange::getAllPos()},
                [this, h, w, f, r, s, cpg, dr]() {
                    auto &pos = dr.getBegin();
                    auto nn = pos[0], hh = pos[1], ww = pos[2], cc = pos[3];
                    auto input = inputs[0], weight = inputs[1],
                         output = outputs[0];
                    VType val = 0;
                    for (int ff = 0; ff < f; ff++)
                        for (int rr = 0; rr < r; rr++)
                            for (int ss = 0; ss < s; ss++) {
                                int ah = ((h + 2 * ph - r) % sh);
                                int aw = ((w + 2 * pw - s) % sw);
                                if (r - ah < rr || s - aw < ss) {
                                    continue;
                                }
                                int posH = hh + rr * dh - (r - ph - 1);
                                int posW = ww + ss * dw - (s - pw - 1);
                                if (posH % sh == 0 && posW % sw == 0) {
                                    posH /= sh;
                                    posW /= sw;
                                    if (posH < 0 || posH >= h || posW < 0 ||
                                        posW >= w) {
                                        continue;
                                    }
                                    VType inputVal =
                                        input->getData({nn, posH, posW, ff});
                                    VType weightVal = weight->getData(
                                        {(r - 1 - rr), (s - 1 - ss), ff, cc});
                                    val += weightVal * inputVal;
                                }
                            }
                    output->dataMalloc();
                    return output->setData({nn, hh, ww, cc}, val);
                }};
    }
}

// input {n, h, w, f} weight {r, s, f, c} output {n, h, w, c}
Dim ConvTransOp::computeShape() {
    auto input = inputs[0], weight = inputs[1], output = outputs[0];
    auto n = input->getDims()[0];
    auto h = input->getDims()[1];
    auto w = input->getDims()[2];
    [[maybe_unused]] auto f = input->getDims()[3];
    auto r = weight->getDims()[0];
    auto s = weight->getDims()[1];
    auto c = weight->getDims()[3];
    assert(f == weight->getDims()[2]);
    int on = n, oc = c;
    int oh = 0, ow = 0;
    oh = (h - 1) * sh - 2 * ph + dh * (r - 1) + oph + 1;
    ow = (w - 1) * sw - 2 * pw + dw * (s - 1) + opw + 1;
    auto ret = {on, oh, ow, oc};
    output->setDims(ret);
    output->setType(Tensor::Input);
    return ret;
}

Dim ConvTransOp::computeOutputPenalty(const Dim &p) {
    assert(false);
    assert(p.size() == 4);
    auto np = p[0], hp = p[2], wp = p[3];
    auto input = inputs[0], weight = inputs[1], output = outputs[0];
    auto n = input->getDims()[0] + np;
    auto h = input->getDims()[2] + hp;
    auto w = input->getDims()[3] + wp;
    auto f = weight->getDims()[0];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    int on = n, oc = f;
    int oh = 0, ow = 0;
    oh = (h - 1) * sh - 2 * ph + dh * (r - 1) + oph + 1;
    ow = (w - 1) * sw - 2 * pw + dw * (s - 1) + opw + 1;
    auto outDim = output->getDims();
    return {on - outDim[0], oc - outDim[1], oh - outDim[2], ow - outDim[3]};
}

// Only called by constructors which explicitly set padding size
// computeShape() is called in constructor
void ConvTransOp::setPaddingMode() {
    auto iDim = inputs[0]->getDims();
    auto oDim = outputs[0]->getDims();
    if (iDim[1] == oDim[1] && iDim[2] == oDim[2])
        padding = Same;
    else if (ph == 0 && pw == 0)
        padding = Valid;
}

// TODO: check correctness
// input {n, h, w, f} weight {r, s, f, c}
bool ConvTransOp::checkValid(const TensorVec &inputs) {
    auto input = inputs[0], weight = inputs[1];
    assert(input != nullptr && weight != nullptr);
    // TODO: group trans_conv is not supported by now
    assert(input->getDims()[3] == weight->getDims()[2]);
    // TODO: dilated trans_conv is not supported by now
    assert(dh == 1 && dw == 1);
    if (input->getType() != Tensor::Input ||
        weight->getType() != Tensor::Weight)
        return false;
    if (input->getDims().size() != 4 || weight->getDims().size() != 4)
        return false;
    return true;
}

double ConvTransOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    constexpr int N_ALGO = 7;
    constexpr cudnnConvolutionBwdDataAlgo_t ALGOS[N_ALGO] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT};

    auto input = inputs[0], weight = inputs[1];
    auto n = input->getDims()[0] + pe->withPenalty() * input->getPenalty()[0];
    auto h = input->getDims()[1] + pe->withPenalty() * input->getPenalty()[1];
    auto w = input->getDims()[2] + pe->withPenalty() * input->getPenalty()[2];
    auto f = input->getDims()[3] + pe->withPenalty() * input->getPenalty()[3];
    auto r = weight->getDims()[0];
    auto s = weight->getDims()[1];
    auto c = weight->getDims()[3];
    auto cpg = weight->getDims()[3];
    // FIXME: group cannot be inferred from input and weight for convTrans
    auto g = c / cpg;
    assert(g == 1);

    ConvArgs args = getArgs(pe->withPenalty());

    if (pe->checkOpPerf(ConvTrans, args)) {
        return pe->getOpPerf(ConvTrans, args);
    }

    int channelsPerGrp = cpg; //, channels = c;

    // get inputs
    cudnnTensorDescriptor_t inDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&inDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT, n, f, h, w));

    float *inData;
    inData = pe->getInputPtr();

    // get kernels
    cudnnFilterDescriptor_t knDesc;
    checkCudnnError(cudnnCreateFilterDescriptor(&knDesc));
    checkCudnnError(cudnnSetFilter4dDescriptor(
        knDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, f, channelsPerGrp, r, s));

    float *knData;
    knData = pe->getWeightPtr();

    // get bias
    cudnnTensorDescriptor_t biasDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&biasDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT, 1, c, 1, 1));

    // float *biasData;
    // biasData = pe->getBiasPtr();

    // get convlution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    checkCudnnError(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCudnnError(cudnnSetConvolution2dDescriptor(
        convDesc, ph, pw, sh, sw, dh, dw, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    if (g > 1) {
        checkCudnnError(cudnnSetConvolutionGroupCount(convDesc, g));
    }

    // get activation descriptor
    cudnnActivationDescriptor_t actDesc;
    checkCudnnError(cudnnCreateActivationDescriptor(&actDesc));
    // NOT_PROPAGATE_NAN is requierd by cudnnConvolotionBiasActivationForward
    switch (act) {
    case Relu:
        checkCudnnError(cudnnSetActivationDescriptor(
            actDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
        break;
    case Sigmoid:
        checkCudnnError(cudnnSetActivationDescriptor(
            actDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
        break;
    case None:
        checkCudnnError(cudnnSetActivationDescriptor(
            actDesc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0));
        break;
    default:
        assert(false);
    }

    // get outputs
    int outn = outputs[0]->getDims()[0], outc = outputs[0]->getDims()[3],
        outh = outputs[0]->getDims()[1], outw = outputs[0]->getDims()[2];
    // checkCudnnError(cudnnGetConvolution2dForwardOutputDim(
    // convDesc, inDesc, knDesc, &outn, &outc, &outh, &outw));
    cudnnTensorDescriptor_t outDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&outDesc));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outn, outc, outh, outw));

    float *outData;
    outData = pe->getOutputPtr();

    ConvTransResult best;
    best.time = INFINITY;
    for (int i = 0; i < N_ALGO; i++) {
        // get workspace
        size_t wsSize;
        auto stat = cudnnGetConvolutionBackwardDataWorkspaceSize(
            pe->cudnnHandle(), knDesc, inDesc, convDesc, outDesc, ALGOS[i],
            &wsSize);
        if (stat != CUDNN_STATUS_SUCCESS) {
            continue;
        }
        // assert(wsSize < (size_t)3 * 1024 * 1024 * 1024);
        if (wsSize >= (size_t)10 * 1024 * 1024 * 1024)
            continue;
        float *wsData;
        wsData = pe->getWorkspace();

        // perform convolution
        double durtime = 0.0;
        float alpha = 1.f, beta = 0.f;
        ch::time_point<ch::high_resolution_clock, ch::nanoseconds> beg, end;
        for (int j = 0; j < rounds + warmupRounds; ++j) {
            cudnnStatus_t stat;
            if (act == None || bias == nullptr) {
                checkCudaError(cudaDeviceSynchronize());
                beg = ch::high_resolution_clock::now();
                stat = cudnnConvolutionBackwardData(
                    pe->cudnnHandle(), &alpha, knDesc, knData, inDesc, inData,
                    convDesc, ALGOS[i], wsData, wsSize, &beta, outDesc,
                    outData);
                checkCudaError(cudaDeviceSynchronize());
                end = ch::high_resolution_clock::now();
            } else {
                // TODO: Bias not supported by now
                assert(false);
                // checkCudaError(cudaDeviceSynchronize());
                // beg = ch::high_resolution_clock::now();
                // stat = cudnnConvolutionBiasActivationBackwardData(
                //     pe->cudnnHandle(), &alpha, inDesc, inData, knDesc,
                //     knData, convDesc, ALGOS[i], wsData, wsSize, &beta,
                //     outDesc, outData, biasDesc, biasData, actDesc, outDesc,
                //     outData);
                // checkCudaError(cudaDeviceSynchronize());
                // end = ch::high_resolution_clock::now();
            }
            if (stat != CUDNN_STATUS_SUCCESS) {
                durtime = INFINITY;
                break;
            }
            if (j >= warmupRounds) {
                durtime +=
                    ch::duration_cast<ch::duration<double>>(end - beg).count() *
                    1000; // ms
            }
        }
        durtime /= rounds;
        if (durtime < best.time) {
            best = ConvTransResult{durtime, ALGOS[i], wsSize};
        }
        // std::cout << "[perf] " << ALGOS[i] << ", " << durtime << std::endl;
    }

    // finalize
    checkCudnnError(cudnnDestroyTensorDescriptor(inDesc));
    checkCudnnError(cudnnDestroyTensorDescriptor(outDesc));
    checkCudnnError(cudnnDestroyFilterDescriptor(knDesc));
    checkCudnnError(cudnnDestroyConvolutionDescriptor(convDesc));

    pe->saveOpPerf(ConvTrans, args, best);

    return best.time;
}

bool ConvTransOp::same(const ConvTransOp &rhs) {
    bool ret = true;
    if (padding == Other)
        ret &= ph == rhs.ph && pw == rhs.pw;
    else
        ret &= padding == rhs.padding;
    ret &= sh == rhs.sh && sw == rhs.sw;
    ret &= dh == rhs.dh && dw == rhs.dw;
    return ret;
}

// TODO: may wrong
void ConvTransOp::inferSplittingPoints() {
    auto inputSplittingPoints = getInputs()[0]->getSplittingPoints();
    // : Operator(Conv, {input, weight}, {output}), ph(ph), pw(pw), sh(sh),
    // sw(sw),
    //   dh(dh), dw(dw), bias(bias), act(act) {
    assert(inputSplittingPoints->size() == 4);
    for (auto &points : *inputSplittingPoints)
        assert(points.empty());

    SplittingPoints splittingPoints(4);
    int h = inputs[0]->getDims()[2], w = inputs[0]->getDims()[3];
    int kh = inputs[1]->getDims()[2], kw = inputs[1]->getDims()[3];

    auto insert = [](std::vector<int> &points, int h, int kh, int ph, int sh,
                     int dh) {
        int last_left_point = (ph + sh - 1) / sh;
        // int first_right_point = (h - 1 - dh * (kh - 1)) / sh + 1;
        int first_right_point = (ph + h - dh * (kh - 1) + sh - 1) / sh;
        for (int i = 1; i <= last_left_point; ++i)
            points.emplace_back(i);
        // omit the boudary point
        for (int i = first_right_point; i < h; ++i)
            points.emplace_back(i);
    };
    insert(splittingPoints[2], h, kh, ph, sh, dh);
    insert(splittingPoints[3], w, kw, pw, sw, dw);
    getOutput()->setSplittingPoints(std::move(splittingPoints));
}

G2BMMOp::G2BMMOp(Tensor *A, Tensor *B, int width, int dilation, Tensor *bias,
                 ActType act)
    : Operator(G2BMM, {A, B}, {}), width(width), dilation(dilation), bias(bias),
      act(act) {
    checkAndSetTensorTypeForConstructor(A, B);
    assert(checkValid({A, B}));
    outputs.emplace_back(new Tensor);
    computeShape();
    initHash();
}

G2BMMOp::G2BMMOp(int width, int dilation, Tensor *bias, ActType act)
    : Operator(G2BMM), width(width), dilation(dilation), bias(bias), act(act) {
    initHash();
}

void G2BMMOp::initHash() {
    hash = type;
    const auto &[b, m, k, width, dilation] = getArgs();
    hash = hashAppend(hash, b);
    hash = hashAppend(hash, m);
    hash = hashAppend(hash, k);
    hash = hashAppend(hash, width);
    hash = hashAppend(hash, dilation);
    hash = hashPack(hash);
}

// This function will only be called by constructor to set the tensor type
// TODO: this function can be removed when more tensor types are introduced
void G2BMMOp::checkAndSetTensorTypeForConstructor(Tensor *A, Tensor *B) {
    if (A->getType() == Tensor::Input && B->getType() == Tensor::Input)
        B->setType(Tensor::Weight);
}

// TODO: fill compute
Tensor *G2BMMOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    // auto A = inputs[0], B = inputs[1], C = outputs[0];
    auto C = outputs[0];
    C->dataMalloc();
    C->setComputed();
    return C;
}

// TODO: is needed?
void G2BMMOp::dimExtend(Tensor *t) {
    assert(false);
    // auto dm = t->getDims();
    // if (dm.size() == 2) {
    //     dm.insert(dm.begin(), 1);
    //     t->setDims(dm);
    // }
}

// TODO: fill compute
std::pair<std::vector<DimRange>, std::function<bool()>>
G2BMMOp::compute(DimRange dr) {
    return {};
}

bool G2BMMOp::checkValid(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    if (A->getType() == Tensor::Weight && B->getType() == Tensor::Weight)
        return false;
    if (A->getDims().size() != 3 || B->getDims().size() != 3) {
        return false;
    }
    if (A->getDims()[0] != B->getDims()[0]) {
        return false;
    }
    if (A->getDims()[1] != B->getDims()[1]) {
        return false;
    }
    if (A->getDims()[2] != B->getDims()[2]) {
        return false;
    }
    if (width < 0) {
        return false;
    }
    return true;
}

Dim G2BMMOp::computeShape() {
    auto A = inputs[0], C = outputs[0];
    auto b = A->getDims()[0];
    auto m = A->getDims()[1];
    auto ret = {b, m, 2 * width + 1};
    C->setDims(ret);
    C->setType(Tensor::Input);
    return ret;
}

double G2BMMOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    auto A = inputs[0];
    auto bs = A->getDims()[0];
    auto n = A->getDims()[1];
    auto m = A->getDims()[2];

    G2BMMGBMMLArgs args = getArgs();
    if (pe->checkOpPerf(G2BMM, args)) {
        return pe->getOpPerf(G2BMM, args);
    }

    float *dA, *dB, *dC;
    dA = pe->getMatA();
    dB = pe->getMatB();
    dC = pe->getMatC();

    // G2BMM cost too long
    if (A->getDims()[0] > 100) {
        warmupRounds = 1;
        rounds = 3;
    }

    for (int i = 0; i < warmupRounds; ++i) {
        _sg2bmm(dA, dB, dC, bs, n, m, width, dilation);
    }
    checkCudaError(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < rounds; ++i) {
        _sg2bmm(dA, dB, dC, bs, n, m, width, dilation);
    }
    cudaEventRecord(stop);
    checkCudaError(cudaDeviceSynchronize());
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= rounds;
    pe->saveOpPerf(G2BMM, args, milliseconds);
    return milliseconds;
}

// TODO: splitting points
void G2BMMOp::inferSplittingPoints() { assert(false); }

GBMMLOp::GBMMLOp(Tensor *A, Tensor *B, int dilation, Tensor *bias, ActType act)
    : Operator(GBMML, {A, B}, {}), dilation(dilation), bias(bias), act(act) {
    checkAndSetTensorTypeForConstructor(A, B);
    assert(checkValid({A, B}));
    outputs.emplace_back(new Tensor);
    computeShape();
    initHash();
}

GBMMLOp::GBMMLOp(int dilation, Tensor *bias, ActType act)
    : Operator(GBMML), dilation(dilation), bias(bias), act(act) {
    initHash();
}

void GBMMLOp::initHash() {
    hash = type;
    const auto &[b, m, w, n, dilation] = getArgs();
    hash = hashAppend(hash, b);
    hash = hashAppend(hash, m);
    hash = hashAppend(hash, w);
    hash = hashAppend(hash, n);
    hash = hashAppend(hash, dilation);
    hash = hashPack(hash);
}

// This function will only be called by constructor to set the tensor type
// TODO: this function can be removed when more tensor types are introduced
void GBMMLOp::checkAndSetTensorTypeForConstructor(Tensor *A, Tensor *B) {
    // Both of inputs tensor are Input type in attention
    // if (A->getType() == Tensor::Input && B->getType() == Tensor::Input)
    //     B->setType(Tensor::Weight);
}

// TODO: fill compute
Tensor *GBMMLOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    // auto A = inputs[0], B = inputs[1], C = outputs[0];
    auto C = outputs[0];
    C->dataMalloc();
    C->setComputed();
    return C;
}

// TODO: is needed?
void GBMMLOp::dimExtend(Tensor *t) {
    assert(false);
    // auto dm = t->getDims();
    // if (dm.size() == 2) {
    //     dm.insert(dm.begin(), 1);
    //     t->setDims(dm);
    // }
}

// TODO: fill compute
std::pair<std::vector<DimRange>, std::function<bool()>>
GBMMLOp::compute(DimRange dr) {
    return {};
}

bool GBMMLOp::checkValid(const TensorVec &inputs) {
    auto A = inputs[0], B = inputs[1];
    if (A->getType() == Tensor::Weight && B->getType() == Tensor::Weight)
        return false;
    if (A->getDims().size() != 3 || B->getDims().size() != 3) {
        return false;
    }
    if (A->getDims()[0] != B->getDims()[0]) {
        return false;
    }
    if (A->getDims()[1] != B->getDims()[1]) {
        return false;
    }
    // TODO: is sufficient?
    if (A->getDims()[2] % 2 == 0) {
        return false;
    }
    return true;
}

Dim GBMMLOp::computeShape() {
    auto A = inputs[0], B = inputs[1], C = outputs[0];
    auto b = A->getDims()[0];
    auto m = A->getDims()[1];
    auto k = B->getDims()[2];
    auto ret = {b, m, k};
    C->setDims(ret);
    C->setType(Tensor::Input);
    return ret;
}

double GBMMLOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    auto A = inputs[0], B = inputs[1];
    auto bs = A->getDims()[0];
    auto n = A->getDims()[1];
    auto w = (A->getDims()[2] - 1) / 2;
    auto m = B->getDims()[2];

    G2BMMGBMMLArgs args = getArgs();
    if (pe->checkOpPerf(GBMML, args)) {
        return pe->getOpPerf(GBMML, args);
    }

    float *dA, *dB, *dC;
    dA = pe->getMatA();
    dB = pe->getMatB();
    dC = pe->getMatC();

    for (int i = 0; i < warmupRounds; ++i) {
        _sgbmml(dA, dB, dC, bs, n, m, w, dilation);
    }
    checkCudaError(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < rounds; ++i) {
        _sgbmml(dA, dB, dC, bs, n, m, w, dilation);
    }
    cudaEventRecord(stop);
    checkCudaError(cudaDeviceSynchronize());
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= rounds;
    pe->saveOpPerf(GBMML, args, milliseconds);
    return milliseconds;
}

// TODO: splitting points
void GBMMLOp::inferSplittingPoints() { assert(false); }

PadOp::PadOp(Tensor *input, Tensor *output, const Dim &begin, const Dim &end)
    : Operator(Pad, {input}, {output}), begin(begin), end(end) {
    assert(checkValid({input}));
    if (output->getDims().size() == 0)
        computeShape();
    initHash();
}

PadOp::PadOp(Tensor *input, const Dim &begin, const Dim &end)
    : Operator(Pad, {input}, {}), begin(begin), end(end) {
    assert(checkValid({input}));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

PadOp::PadOp(const Dim &begin, const Dim &end)
    : Operator(Pad), begin(begin), end(end) {
    initHash();
}

PadOp::PadOp(const PadOp &rhs)
    : Operator(rhs), begin(rhs.begin), end(rhs.end) {}

void PadOp::initHash() {
    hash = type;
    hash = hashAppend(hash, begin.size());
    for (auto x : begin) {
        hash = hashAppend(hash, x);
    }
    hash = hashAppend(hash, end.size());
    for (auto x : end) {
        hash = hashAppend(hash, x);
    }
    hash = hashPack(hash);
}

Tensor *PadOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    size_t iEnd = outputs[0]->size();
    const Dim &inDim = inputs[0]->getDims();
    const Dim &outDim = outputs[0]->getDims();
#pragma omp parallel for
    for (size_t i = 0; i < iEnd; i++) {
        Dim outIdx = cntToIdx(outDim, i);
        Dim inIdx = elementwiseSub(outIdx, begin);
        for (size_t j = 0, jEnd = outIdx.size(); j < jEnd; j++) {
            if (inIdx[j] < 0 || inIdx[j] >= inDim[j]) {
                outputs[0]->setData(outIdx, 0);
                continue;
            }
        }
        outputs[0]->setData(outIdx, inputs[0]->getData(inIdx));
    }

    outputs[0]->setComputed();
    return outputs[0];
}

std::pair<std::vector<DimRange>, std::function<bool()>>
PadOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    // TODO: dr should not fall into zero areas
    return {{DimRange::getAllPos()}, [this]() { return compute() != nullptr; }};
}

Dim PadOp::computeShape() {
    Dim dim = inputs[0]->getDims();
    for (size_t i = 0, iEnd = dim.size(); i < iEnd; i++)
        dim[i] += begin[i] + end[i];
    outputs[0]->setDims(dim);
    outputs[0]->setType(inputs[0]->getType());
    return dim;
}

bool PadOp::checkValid(const TensorVec &inputs) {
    const Dim &dim = inputs[0]->getDims();
    if (begin.size() != dim.size() || end.size() != dim.size())
        return false;
    for (size_t i = 0, iEnd = dim.size(); i < iEnd; i++)
        if (begin[i] < 0 || end[i] < 0)
            return false;
    return true;
}

double PadOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    // assert(false);
    return 0;
}

std::string PadOp::toString() const {
    std::ostringstream os;
    os << "Pad(in=" << inputs[0]->getHash() << ",out=" << outputs[0]->getHash()
       << ")";
    return os.str();
}

SliceOp::SliceOp(Tensor *input, Tensor *output, const Dim &begin,
                 const Dim &end)
    : Operator(Slice, {input}, {output}), begin(begin), end(end) {
    assert(checkValid({input}));
    initHash();
}

SliceOp::SliceOp(Tensor *input, const Dim &begin, const Dim &end)
    : Operator(Slice, {input}, {}), begin(begin), end(end) {
    assert(checkValid({input}));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

SliceOp::SliceOp(const Dim &begin, const Dim &end)
    : Operator(Slice), begin(begin), end(end) {
    initHash();
}

SliceOp::SliceOp(const SliceOp &rhs)
    : Operator(rhs), begin(rhs.begin), end(rhs.end) {}

void SliceOp::initHash() {
    hash = type;
    hash = hashAppend(hash, begin.size());
    for (auto x : begin) {
        hash = hashAppend(hash, x);
    }
    hash = hashAppend(hash, end.size());
    for (auto x : end) {
        hash = hashAppend(hash, x);
    }
    hash = hashPack(hash);
}

Tensor *SliceOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    size_t iEnd = outputs[0]->size();
    const Dim &outDim = outputs[0]->getDims();
#pragma omp parallel for
    for (size_t i = 0; i < iEnd; i++) {
        Dim outIdx = cntToIdx(outDim, i);
        Dim inIdx = elementwiseAdd(outIdx, begin);
        outputs[0]->setData(outIdx, inputs[0]->getData(inIdx));
    }

    outputs[0]->setComputed();
    return outputs[0];
}

std::pair<std::vector<DimRange>, std::function<bool()>>
SliceOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    return {{DimRange::getAllPos()}, [this]() { return compute() != nullptr; }};
}

Dim SliceOp::computeShape() {
    Dim dim = inputs[0]->getDims();
    for (size_t i = 0, iEnd = dim.size(); i < iEnd; i++)
        dim[i] -= begin[i] + end[i];
    outputs[0]->setDims(dim);
    outputs[0]->setType(inputs[0]->getType());
    return dim;
}

bool SliceOp::checkValid(const TensorVec &inputs) {
    const Dim &dim = inputs[0]->getDims();
    if (begin.size() != dim.size() || end.size() != dim.size())
        return false;
    for (size_t i = 0, iEnd = dim.size(); i < iEnd; i++)
        if (begin[i] < 0 || end[i] < 0 || dim[i] < begin[i] + end[i])
            return false;
    return true;
}

double SliceOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    // assert(false);
    return 0;
}

std::string SliceOp::toString() const {
    std::ostringstream os;
    os << "Slice(in=" << inputs[0]->getHash()
       << ",out=" << outputs[0]->getHash() << ")";
    return os.str();
}

ConcatOp::ConcatOp(TensorVec inputs, int dim)
    : Operator(Concat, inputs, {}), dim(dim) {
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

ConcatOp::ConcatOp(int dim) : Operator(Concat), dim(dim) { initHash(); }

void ConcatOp::initHash() {
    hash = type;
    hash = hashAppend(hash, dim);
    hash = hashPack(hash);
}

Tensor *ConcatOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto output = outputs[0];
    output->dataMalloc();
    std::vector<Dim> iDims;
    for (auto input : inputs)
        iDims.emplace_back(input->getDims());
    auto oDim = output->getDims();
    size_t blockOffsetInner = 1;
    for (size_t i = oDim.size() - 1; i > (size_t)dim; --i)
        blockOffsetInner *= oDim[i];
    size_t blockOffset = oDim[dim] * blockOffsetInner;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input = inputs[i];
        auto dimOffset = 0;
        auto iDim = iDims[i];
        for (size_t j = 0; j < i; ++j)
            dimOffset += iDims[j][dim];
        size_t localBlockOffset = 1;
        for (size_t i = iDim.size() - 1; i >= (size_t)dim && i != (size_t)-1;
             --i)
            localBlockOffset *= iDim[i];
        auto innerOffset = blockOffsetInner * dimOffset;
        auto iSz = input->size();
        auto i_ptr = input->getDataPtr(), o_ptr = output->getDataPtr();
#pragma omp parallel for
        for (size_t iOffset = 0; iOffset < iSz; ++iOffset) {
            auto oOffset = iOffset % localBlockOffset + innerOffset +
                           iOffset / localBlockOffset * blockOffset;
            // output->setData(oOffset, input->getData(iOffset));
            o_ptr[oOffset] = i_ptr[iOffset];
        }
    }
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
ConcatOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {std::vector<DimRange>(inputs.size(), DimRange::getEmpty()),
                []() { return true; }};
    if (!dr.isSinglePos())
        return {std::vector<DimRange>(inputs.size(), DimRange::getAllPos()),
                [this]() { return compute() != nullptr; }};
    auto pos = dr.getBegin();
    int dimSum = 0;
    size_t idx = 0;
    for (auto numInputs = inputs.size(); idx < numInputs; ++idx) {
        dimSum += inputs[idx]->getDims()[dim];
        if (pos[dim] < dimSum) {
            dimSum -= inputs[idx]->getDims()[dim];
            break;
        }
    }
    auto inputPos = pos;
    inputPos[dim] = inputPos[dim] - dimSum;
    auto ret = std::vector<DimRange>(inputs.size(), DimRange::getEmpty());
    ret[idx] = inputPos;
    return {ret, [this, pos, idx, inputPos]() {
                outputs[0]->dataMalloc();
                return outputs[0]->setData(pos, inputs[idx]->getData(inputPos));
            }};
}

bool ConcatOp::checkValid(const TensorVec &tensors) {
    if (tensors.empty())
        return false;
    auto outputDims = tensors[0]->getDims();
    int concatDim = 0;
    for (auto input : tensors) {
        assert(input != nullptr);
        if (input->getType() != tensors[0]->getType())
            return false;
        auto inputDims = input->getDims();
        if (inputDims.size() != outputDims.size())
            return false;
        for (size_t i = 0; i < inputDims.size(); ++i)
            if (i != (size_t)dim) {
                if (inputDims[i] != outputDims[i])
                    return false;
                else
                    concatDim += inputDims[i];
            }
    }
    return true;
}

void ConcatOp::inferSplittingPoints() {
    // Wihtout enough test
    SplittingPoints points(inputs[0]->getDims().size());
    for (int i = 0; i < (int)inputs[0]->getDims().size(); ++i) {
        auto &input0 = (*inputs[0]->getSplittingPoints())[i];
        auto &input1 = (*inputs[1]->getSplittingPoints())[i];

        if (i == dim) {
            points[i].insert(points[i].begin(), input0.begin(), input0.end());
            points[i].emplace_back(inputs[0]->getDims()[i]);
            for (int v : input1)
                points[i].emplace_back(v + inputs[0]->getDims()[i]);
        } else {
            std::merge(input0.begin(), input0.end(), input1.begin(),
                       input1.end(), std::back_inserter(points[i]));
        }
    }
    outputs[0]->setSplittingPoints(points);
}

Dim ConcatOp::computeShape() {
    auto ret = inputs[0]->getDims();
    ret[dim] = 0;
    for (auto input : inputs)
        ret[dim] += input->getDims()[dim];
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

SplitOp::SplitOp(Tensor *input, int dim, int num)
    : Operator(Split, {input}, {}), dim(dim), num(num), sizes({}) {
    assert(input != nullptr);
    for (int i = 0; i < num; ++i)
        outputs.emplace_back(new Tensor());
    computeShapeV();
    initHash();
}

SplitOp::SplitOp(Tensor *input, int dim, const std::vector<int> &sizes)
    : Operator(Split, {input}, {}), dim(dim), num(-1), sizes(sizes) {
    assert(input != nullptr);
    for (size_t i = 0; i < sizes.size(); ++i)
        outputs.emplace_back(new Tensor());
    computeShapeV();
    initHash();
}

SplitOp::SplitOp(int dim, int num)
    : Operator(Split), dim(dim), num(num), sizes({}) {
    initHash();
}

SplitOp::SplitOp(int dim, const std::vector<int> &sizes)
    : Operator(Split), dim(dim), num(-1), sizes(sizes) {
    initHash();
}

void SplitOp::initHash() {
    hash = type;
    hash = hashAppend(hash, dim);
    hash = hashAppend(hash, num);
    hash = hashAppend(hash, sizes.size());
    for (auto x : sizes) {
        hash = hashAppend(hash, x);
    }
    hash = hashPack(hash);
}

Tensor *SplitOp::compute() {
    assert(false); // Not supported
}

TensorVec SplitOp::computeV() {
    auto allComputed = true;
    for (auto output : outputs)
        allComputed &= output->isComputed();
    if (allComputed)
        return outputs;

    auto input = inputs[0];
    auto &iDim = input->getDims();
    std::vector<Dim> oDims;
    for (auto output : outputs)
        oDims.emplace_back(output->getDims());
    size_t blockOffsetInner = 1;
    for (size_t i = iDim.size() - 1; i > (size_t)dim; --i)
        blockOffsetInner *= iDim[i];
    size_t blockOffset = iDim[dim] * blockOffsetInner;
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        output->dataMalloc();
        auto dimOffset = 0;
        auto oDim = oDims[i];
        for (size_t j = 0; j < i; ++j)
            dimOffset += oDims[j][dim];
        size_t localBlockOffset = 1;
        for (size_t i = oDim.size() - 1; i >= (size_t)dim && i != (size_t)-1;
             --i)
            localBlockOffset *= oDim[i];
        auto innerOffset = blockOffsetInner * dimOffset;
        auto oSz = output->size();
        auto i_ptr = input->getDataPtr(), o_ptr = output->getDataPtr();
#pragma omp parallel for
        for (size_t oOffset = 0; oOffset < oSz; ++oOffset) {
            auto iOffset = oOffset % localBlockOffset + innerOffset +
                           oOffset / localBlockOffset * blockOffset;
            // output->setData(oOffset, input->getData(iOffset));
            o_ptr[oOffset] = i_ptr[iOffset];
        }
    }
    for (auto output : outputs)
        output->setComputed();

    return outputs;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
SplitOp::compute(DimRange dr) {
    assert(false); // Not supported
}

std::pair<std::vector<DimRange>, std::function<bool()>>
SplitOp::compute(size_t idx, DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    if (!dr.isSinglePos())
        return {{DimRange::getAllPos()},
                [this]() { return !computeV().empty(); }};
    auto pos = dr.getBegin();
    auto inputPos = pos;
    for (size_t i = 0; i < idx; ++i) {
        inputPos[dim] += outputs[i]->getDims()[dim];
    }
    return {{DimRange(inputPos)}, [this, pos, idx, inputPos]() {
                outputs[idx]->dataMalloc();
                return outputs[idx]->setData(pos, inputs[0]->getData(inputPos));
            }};
}

bool SplitOp::compute(const TensorVec &inputTensors,
                      const TensorVec &outputTensors) {
    if (computeShape(inputTensors, outputTensors) && !computeV().empty())
        return true;
    return false;
}

Dim SplitOp::computeShape() {
    assert(false);
    return Dim{};
}

bool SplitOp::computeShape(const TensorVec &inputTensors,
                           const TensorVec &outputTensors) {
    if (inputTensors.size() != 1 || (num == -1 && outputTensors.empty()))
        return false;
    auto input = inputTensors[0];
    if (input->getDims().size() <= (size_t)dim)
        return false;
    this->inputs = inputTensors;
    this->outputs = outputTensors;
    auto dimVec = computeShapeV();
    if (dimVec.empty())
        return false;
    return true;
}

std::vector<Dim> SplitOp::computeShapeV() {
    std::vector<Dim> ret;
    if (sizes.empty()) {
        if (num == -1) {
            for (auto output : outputs)
                output->setInvalid();
            return {};
        } else
            setSizesForEqualySplit();
    }
    auto input = inputs[0];
    int total = std::accumulate(sizes.begin(), sizes.end(), 0);
    int splitDimSz = input->getDims()[dim];
    if (splitDimSz % total != 0) {
        for (auto output : outputs)
            output->setInvalid();
        return {};
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto outDim = input->getDims();
        outDim[dim] = splitDimSz / total * sizes[i];
        ret.emplace_back(outDim);
        outputs[i]->setDims(outDim);
        if (outputs[i]->getType() != Tensor::NotCounted)
            outputs[i]->setType(input->getType());
    }
    return ret;
}

TransposeOp::TransposeOp(Tensor *input, Tensor *output, const Perm &before,
                         const Perm &after, int factor, TransType trans_type)
    : Operator(Transpose, {input}, {output}), before(before), after(after),
      factor(factor), trans_type(trans_type) {
    assert(checkValid({input}));
    initHash();
    split = -1;
    for (size_t i = 0, iEnd = before.size(); i < iEnd; ++i)
        if (!before[i].isSingle()) {
            split = i;
            break;
        }
};

// The index of the splitting introduced dim is -1
TransposeOp::TransposeOp(Tensor *input, Tensor *output, int split,
                         const Perm &after, int factor, TransType trans_type)
    : Operator(Transpose, {input}, {output}), before({}), after(after),
      split(split), factor(factor), trans_type(trans_type) {
    initHash();
    setParam(input->getDims().size());
    assert(checkValid({input}));
    computeShape();
};

TransposeOp::TransposeOp(Tensor *input, int split, const Perm &after,
                         int factor, TransType trans_type)
    : Operator(Transpose, {input}, {}), before({}), after(after), split(split),
      factor(factor), trans_type(trans_type) {
    initHash();
    setParam(input->getDims().size());
    assert(checkValid({input}));
    outputs.emplace_back(new Tensor());
    computeShape();
}

TransposeOp::TransposeOp(int split, const Perm &after, int factor,
                         TransType trans_type)
    : Operator(Transpose), before({}), after(after), split(split),
      factor(factor), trans_type(trans_type) {
    initHash();
    int dimSz = 0;
    for (auto i : after.asVector())
        dimSz = std::max(dimSz, i);
    dimSz += 1;
    setParam(dimSz);
}

TransposeOp::TransposeOp(const TransposeOp &rhs)
    : Operator(rhs), before(rhs.before), after(rhs.after), split(rhs.split),
      factor(rhs.factor), trans_type(rhs.trans_type) {}

// hash: type|factor|split|merge|after(list)
void TransposeOp::initHash() {
    hash = type;
    hash = hashAppend(hash, factor);
    hash = hashAppend(hash, split);
    int merge = -1;
    for (size_t i = 0, iEnd = after.getPerm().size(); i < iEnd; i++) {
        if (!after.getPerm()[i].isSingle()) {
            merge = i;
        }
    }
    hash = hashAppend(hash, merge);
    for (auto x : after.asVector()) {
        hash = hashAppend(hash, x);
    }
}

Tensor *TransposeOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    outputs[0]->dataMalloc();
    Dim flatDim;
    for (size_t i = 0, iEnd = before.size(); i < iEnd; ++i) {
        if (before[i].isSingle())
            flatDim.emplace_back(inputs[0]->getDims()[i]);
        else {
            if (factor > 0) {
                flatDim.emplace_back(inputs[0]->getDims()[i] / factor);
                if (inputs[0]->getDims()[i] % factor != 0)
                    return nullptr;
                flatDim.emplace_back(factor);
            } else {
                flatDim.emplace_back(-factor);
                flatDim.emplace_back(inputs[0]->getDims()[i] / (-factor));
                if (inputs[0]->getDims()[i] % (-factor) != 0)
                    return nullptr;
            }
        }
    }

    size_t iEnd = inputs[0]->size();
#pragma omp parallel for
    for (size_t i = 0; i < iEnd; ++i) {
        Dim flatIt = cntToIdx(flatDim, i);
        int afterIdx = 0;
        for (int j = 0, jEnd = after.size(); j < jEnd; ++j) {
            auto &pos = after[j].getVec();
            for (size_t k = 0, kEnd = pos.size(); k < kEnd; ++k)
                afterIdx = afterIdx * flatDim[pos[k]] + flatIt[pos[k]];
        }
        outputs[0]->setData(afterIdx, inputs[0]->getData(i));
    }
    outputs[0]->setComputed();
    return outputs[0];
}

std::pair<std::vector<DimRange>, std::function<bool()>>
TransposeOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    if (!dr.isSinglePos()) {
        return {{DimRange::getAllPos()},
                [this]() { return compute() != nullptr; }};
    }
    auto pos = dr.getBegin();
    auto input = inputs[0], output = outputs[0];
    if (pos.size() != input->getDims().size())
        return {};
    for (size_t i = 0, iEnd = pos.size(); i < iEnd; ++i)
        if (pos[i] >= output->getDims()[i])
            return {};

    Dim reshapeDim1;
    for (size_t i = 0, iEnd = before.size(); i < iEnd; ++i) {
        if (before[i].isSingle())
            reshapeDim1.emplace_back(inputs[0]->getDims()[i]);
        else {
            assert(before[i].getVec().size() == 2);
            if (factor > 0) {
                reshapeDim1.emplace_back(inputs[0]->getDims()[i] / factor);
                if (inputs[0]->getDims()[i] % factor != 0)
                    return {};
                reshapeDim1.emplace_back(factor);
            } else {
                reshapeDim1.emplace_back(-factor);
                reshapeDim1.emplace_back(inputs[0]->getDims()[i] / (-factor));
                if (inputs[0]->getDims()[i] % (-factor) != 0)
                    return {};
            }
        }
    }

    // Dim reshapeDim2 = Dim(reshapeDim1.size(), 0);
    // auto beforeVec = before.asVector(), afterVec = after.asVector();
    // assert(beforeVec.size() == afterVec.size());
    // for (size_t i = 0, iEnd = beforeVec.size(); i < iEnd; ++i)
    //     reshapeDim2[i] = reshapeDim1[afterVec[i]];

    Dim afterFlatIt;
    for (size_t i = 0, iEnd = after.size(); i < iEnd; ++i) {
        if (after[i].isSingle())
            afterFlatIt.emplace_back(pos[i]);
        else {
            auto localPerm = after[i].getVec();
            auto localSz = localPerm.size();
            Dim localFlatIt = Dim(localSz, 0);
            auto idx = localSz - 1;
            auto cur = pos[i];
            while (cur > 0) {
                localFlatIt[idx] = cur % reshapeDim1[localPerm[idx]];
                cur /= reshapeDim1[localPerm[idx]];
                idx--;
            }
            for (auto dm : localFlatIt)
                afterFlatIt.emplace_back(dm);
        }
    }
    assert(afterFlatIt.size() == reshapeDim1.size());

    auto afterVec = after.asVector();
    Dim beforeFlatIt = Dim(afterFlatIt.size(), 0);
    for (size_t i = 0, iEnd = afterFlatIt.size(); i < iEnd; ++i)
        beforeFlatIt[afterVec[i]] = afterFlatIt[i];

    Dim inputPos;
    for (size_t i = 0, iEnd = before.size(), j = 0; i < iEnd; ++i) {
        if (before[i].isSingle())
            inputPos.emplace_back(beforeFlatIt[j++]);
        else {
            auto localPerm = before[i].getVec();
            size_t localIdx = 0;
            auto localPos = beforeFlatIt[j + localIdx];
            while (++localIdx < localPerm.size()) {
                localPos = localPos * reshapeDim1[j + localIdx] +
                           beforeFlatIt[j + localIdx];
            }
            j += localPerm.size();
            inputPos.emplace_back(localPos);
        }
    }

    return {{DimRange(inputPos)}, [this, pos, inputPos]() {
                outputs[0]->dataMalloc();
                return outputs[0]->setData(pos, inputs[0]->getData(inputPos));
            }};
}

bool TransposeOp::checkValid(const TensorVec &inputs) {
    if (split != -1 && factor == 0)
        return false;
    // TODO: check this
    // if (factor > -2 && factor < 2)
    //    return false;
    auto beforeVec = before.asVector();
    auto afterVec = after.asVector();
    if (beforeVec.size() != afterVec.size())
        return false;
    std::sort(afterVec.begin(), afterVec.end());

    if (before.size() != inputs[0]->getDims().size())
        return false;

    for (size_t i = 0, iEnd = beforeVec.size(); i < iEnd; ++i)
        if (beforeVec[i] != (int)i || afterVec[i] != (int)i)
            return false;

    int cnt = 0;
    for (size_t i = 0, iEnd = before.size(); i < iEnd; ++i)
        if (!before[i].isSingle())
            cnt++;
    if (cnt > 1)
        return false;

    cnt = 0;
    for (size_t i = 0, iEnd = after.size(); i < iEnd; ++i)
        if (!after[i].isSingle())
            cnt++;
    if (cnt > 1)
        return false;
    return true;
}

Dim TransposeOp::computeShape() {
    Dim flatDim;
    for (size_t i = 0, iEnd = before.size(); i < iEnd; ++i) {
        if (before[i].isSingle())
            flatDim.emplace_back(inputs[0]->getDims()[i]);
        else {
            if (factor > 0) {
                // TODO: unequal split?
                if (inputs[0]->getDims()[i] % factor != 0 ||
                    inputs[0]->getDims()[i] < factor) {
                    outputs[0]->setInvalid();
                    return {};
                }
                flatDim.emplace_back(inputs[0]->getDims()[i] / factor);
                flatDim.emplace_back(factor);
            } else {
                // TODO: unequal split?
                if (inputs[0]->getDims()[i] % (-factor) != 0 ||
                    inputs[0]->getDims()[i] < (-factor)) {
                    outputs[0]->setInvalid();
                    return {};
                }
                flatDim.emplace_back(-factor);
                flatDim.emplace_back(inputs[0]->getDims()[i] / (-factor));
            }
        }
    }
    Dim ret = Dim(after.getPerm().size(), 0);
    for (size_t i = 0; i < after.size(); ++i) {
        if (after[i].isSingle())
            ret[i] = flatDim[after[i].getSingle()];
        else {
            ret[i] = 1;
            auto &pitem = after[i].getVec();
            for (size_t j = 0; j < pitem.size(); ++j)
                ret[i] *= flatDim[pitem[j]];
        }
    }
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

double TransposeOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    if (inputs[0]->getType() == Tensor::Weight) {
        return 0;
    } else if (inputs[0]->getOutputOf() != nullptr &&
               inputs[0]->getOutputOf()->getType() == Transpose) {
        return 0;
    } else {
        return inputs[0]->size() * sizeof(float) * 2 / (200.0 * 1024 * 1024);
        // Too large overhead
        // CodeEngine code_engine;
        // code_engine.genTransposeCompute(*this);
        // if (code_engine.last_tvm_op_perf<=0) {
        //     printf("transpose.perf code_engine.last_tvm_op_perf <= 0\n");
        //     return inputs[0]->size() * sizeof(float) * 2 / (400.0 * 1024 *
        //     1024);
        // } else
        //     return code_engine.last_tvm_op_perf;
    }
}

std::vector<std::shared_ptr<TransBasic>> TransposeOp::getTTParam() const {
    int extr = 0, pad = 0;
    int copys = 0, dels = 0;
    switch (trans_type) {
    case N2H:
    case D2H:
        if (trans_pos == Pre)
            pad = padding_h;
        if (trans_pos == Post)
            extr = padding_h;
        break;
    case N2W:
    case D2W:
        if (trans_pos == Pre)
            pad = padding_w;
        if (trans_pos == Post)
            extr = padding_w;
        break;
    case H2N:
        if (trans_pos == Pre)
            copys = padding_h;
        if (trans_pos == Post)
            dels = padding_h;
        break;
    case W2N:
        if (trans_pos == Pre)
            copys = padding_w;
        if (trans_pos == Post)
            dels = padding_w;
        break;
    default:
        break;
    }
    auto ret = std::vector<std::shared_ptr<TransBasic>>();
    auto input = inputs[0];
    auto inputDim = input->getDims();
    int splitFactor = 0;
    if (split >= 0) {
        splitFactor = factor > 0 ? factor : (inputDim[split] / (-factor));
        ret.emplace_back(new TransSplit(split, splitFactor, extr, copys));
    }

    ret.emplace_back(new TransReorder(after.asVector()));

    for (size_t i = 0, iEnd = after.size(); i < iEnd; ++i)
        if (!after[i].isSingle()) {
            for (size_t j = 0, jEnd = after[i].getVec().size() - 1; j < jEnd;
                 j++) {
                ret.emplace_back(new TransFuse(i, pad, dels));
            }
            break;
        }
    return ret;
}

void TransposeOp::inferSplittingPoints() {
    assert(inputs.size() == 1);
    auto const input = inputs[0];
    const auto &inputSplittingPoints = *input->getSplittingPoints();
    assert(before.size() == after.size());
    SplittingPoints intermediatePoints(after.size() + 1);
    SplittingPoints ret(after.size());

    assert(factor == 2 || factor == -2);

    // Splitting an axis
    for (size_t i = 0, iEnd = before.size(); i < iEnd; ++i) {
        if (before[i].isSingle())
            intermediatePoints[before[i].getSingle()] = inputSplittingPoints[i];
        else {
            int k = (factor > 0) ? factor : -factor;
            if (factor > 0) {
                // check unequal split
                assert(inputs[0]->getDims()[i] % k == 0);
                for (int pos : inputSplittingPoints[i])
                    intermediatePoints[before[i].getVec()[0]].emplace_back(pos %
                                                                           k);
            } else {
                assert(inputs[0]->getDims()[i] % (-factor) == 0);
                for (int pos : inputSplittingPoints[i]) {
                    intermediatePoints[before[i].getVec()[1]].emplace_back(pos /
                                                                           k);
                    // splitting points surrounding the original points
                    // to avoid merged boxes. It is sometimes redundent.
                    if (pos % k != 0 &&
                        pos / k + 1 < inputs[0]->getDims()[i] / k)
                        intermediatePoints[before[i].getVec()[1]].emplace_back(
                            pos / k + 1);
                }
            }
        }
    }

    // find the original dim from the before/after index
    auto find_before_index = [this](int k) -> int {
        for (int i = 0; i < (int)this->before.size(); ++i) {
            if (this->before[i].isSingle()) {
                if (this->before[i].getSingle() == k)
                    return i;
            } else if (this->before[i].getVec()[0] == k ||
                       this->before[i].getVec()[1] == k)
                return i;
        }
        assert(0);
        return -1;
    };

    // Merge two axes
    // the new axis of (0..factor)
    int splitKDim = before[split].getVec()[(factor > 0) ? 1 : 0];
    for (size_t i = 0; i < after.size(); ++i) {
        if (after[i].isSingle())
            ret[i] = intermediatePoints[after[i].getSingle()];
        else {
            if (after[i].getVec()[0] == splitKDim) { // (K, W)
                if (intermediatePoints[after[i].getVec()[1]].empty())
                    continue;
                ret[i] = intermediatePoints[after[i].getVec()[1]];
                int origin_size =
                    input->getDims()[find_before_index(after[i].getVec()[1])];
                ret[i].emplace_back(origin_size);
                for (int pos : intermediatePoints[after[i].getVec()[1]])
                    ret[i].emplace_back(origin_size + pos);
            } else if (after[i].getVec()[1] == splitKDim) { // (W, K)
                // interleaving merging
                for (int pos : intermediatePoints[after[i].getVec()[0]]) {
                    ret[i].emplace_back(pos * 2);
                    ret[i].emplace_back(pos * 2 + 1);
                }
            } else
                assert(0);
        }
    }
    // remove duplicates
    for (auto &points : ret) {
        sort(points.begin(), points.end());
        points.erase(unique(points.begin(), points.end()), points.end());
    }
    getOutput()->setSplittingPoints(std::move(ret));
}

std::string TransposeOp::toString() const {
    std::ostringstream os;
    os << "Transpose[" << hash << "] ";
    os << "(before=" << before.toString() << ",";
    os << "after=" << after.toString() << ",";
    os << "split=" << split << ",";
    os << "factor=" << factor << ",";
    os << "input=" << inputs[0]->getHash() << ",";
    os << "output=" << outputs[0]->getHash() << ")";
    os << "input guid=" << inputs[0]->getGuid() << ",";
    os << "output guid=" << outputs[0]->getGuid() << ")";
    return os.str();
}

ExtendOp::ExtendOp(Tensor *input, int dim, int num)
    : Operator(Extend, {input}, {}), dim(dim), num(num) {
    assert(input != nullptr);
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

ExtendOp::ExtendOp(int dim, int num) : Operator(Extend), dim(dim), num(num) {
    initHash();
}

void ExtendOp::initHash() {
    hash = type;
    hash = hashAppend(hash, dim);
    hash = hashAppend(hash, num);
    hash = hashPack(hash);
}

Tensor *ExtendOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto input = inputs[0];
    auto output = outputs[0];
    output->dataMalloc();
    auto oSz = output->size();
    int blockSize = 1;
    auto iDim = input->getDims();
    for (size_t i = iDim.size() - 1; i >= (size_t)dim && i != (size_t)-1; --i)
        blockSize *= iDim[i];
    auto blockSizeOuter = (num + 1) * blockSize;
#pragma omp parallel for
    for (size_t oIdx = 0; oIdx < oSz; ++oIdx) {
        auto iIdx = oIdx % blockSize + oIdx / blockSizeOuter * blockSize;
        output->setData(oIdx, input->getData(iIdx));
    }
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
ExtendOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    if (!dr.isSinglePos())
        return {{DimRange::getAllPos()},
                [this]() { return compute() != nullptr; }};
    auto iDim = inputs[0]->getDims();
    auto pos = dr.getBegin();
    auto inputPos = pos;
    inputPos[dim] %= iDim[dim];
    return {{DimRange(inputPos)}, [this, pos, inputPos]() {
                outputs[0]->dataMalloc();
                return outputs[0]->setData(pos, inputs[0]->getData(inputPos));
            }};
}

Dim ExtendOp::computeShape() {
    auto ret = inputs[0]->getDims();
    ret[dim] *= (num + 1);
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

BatchNormOp::BatchNormOp(Tensor *input, Tensor *scale, Tensor *bias,
                         Tensor *mean, Tensor *var, Tensor *output,
                         float epsilon, float momentum)
    : Operator(BatchNorm, {input}, {output}), epsilon(epsilon),
      momentum(momentum), scale(scale), bias(bias), mean(mean), var(var) {
    // HACK: for convTransposed
    // assert(checkValid({input}));
    scale->setType(Tensor::Weight);
    bias->setType(Tensor::Weight);
    mean->setType(Tensor::Weight);
    var->setType(Tensor::Weight);
    initHash();
}

BatchNormOp::BatchNormOp(Tensor *input, Tensor *scale, Tensor *bias,
                         Tensor *mean, Tensor *var, float epsilon,
                         float momentum)
    : Operator(BatchNorm, {input}, {}), epsilon(epsilon), momentum(momentum),
      scale(scale), bias(bias), mean(mean), var(var) {
    assert(checkValid({input}));
    scale->setType(Tensor::Weight);
    bias->setType(Tensor::Weight);
    mean->setType(Tensor::Weight);
    var->setType(Tensor::Weight);
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

void BatchNormOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

Tensor *BatchNormOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto input = inputs[0];
    auto output = outputs[0];
    output->dataMalloc();
    input->itReset();
    auto c = input->getDims()[1];
    auto vec = std::vector<double>(c, 0);
    for (int cc = 0; cc < c; ++cc)
        vec[cc] = scale->getData(cc) / sqrt(var->getData(cc) + epsilon);
    while (input->itValid()) {
        auto iit = input->itGet();
        auto cc = iit[1];
        VType val = (input->getData(iit) - mean->getData(cc)) * vec[cc] +
                    bias->getData(cc);
        output->setData(iit, val);
    }
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
BatchNormOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    return {{dr}, [this]() { return compute() != nullptr; }};
}

bool BatchNormOp::checkValid(const TensorVec &inputs) {
    auto input = inputs[0];
    assert(input != nullptr && scale != nullptr && bias != nullptr &&
           mean != nullptr && var != nullptr);
    if (input->getType() != Tensor::Input)
        return false;
    if (input->getDims().size() != 4)
        return false;
    if (scale->getDims().size() != 1 || bias->getDims().size() != 1 ||
        mean->getDims().size() != 1 || var->getDims().size() != 1)
        return false;
    auto c = input->getDims()[1];
    if (scale->getDims()[0] != c || bias->getDims()[0] != c ||
        mean->getDims()[0] != c || var->getDims()[0] != c)
        return false;
    return true;
}

Dim BatchNormOp::computeShape() {
    auto ret = inputs[0]->getDims();
    outputs[0]->setDims(ret);
    outputs[0]->setType(Tensor::Input);
    return ret;
}

MaxPoolOp::MaxPoolOp(Tensor *input, int kh, int kw, int dh, int dw, int ph,
                     int pw, int sh, int sw)
    : Operator(MaxPool, {input}, {}), kh(kh), kw(kw), dh(dh), dw(dw), ph(ph),
      pw(pw), sh(sh), sw(sw) {
    outputs.emplace_back(new Tensor());
    initHash();
    computeShape();
}

MaxPoolOp::MaxPoolOp(int kh, int kw, int dh, int dw, int ph, int pw, int sh,
                     int sw)
    : Operator(MaxPool), kh(kh), kw(kw), dh(dh), dw(dw), ph(ph), pw(pw), sh(sh),
      sw(sw) {
    initHash();
}

void MaxPoolOp::initHash() {
    hash = type;
    hash = hashAppend(hash, kh);
    hash = hashAppend(hash, kw);
    hash = hashAppend(hash, dh);
    hash = hashAppend(hash, dw);
    hash = hashAppend(hash, ph);
    hash = hashAppend(hash, pw);
    hash = hashAppend(hash, sh);
    hash = hashAppend(hash, sw);
    hash = hashPack(hash);
}

Tensor *MaxPoolOp::compute() {
    // TODO: maxpool compute
    assert(false);
    return nullptr;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
MaxPoolOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    return {{DimRange::getAllPos()}, [this]() { return compute() != nullptr; }};
}

Dim MaxPoolOp::computeShape() {
    auto input = inputs[0];
    auto h = input->getDims()[input->getDims().size() - 1],
         w = input->getDims()[input->getDims().size() - 2];
    int oh = (h - (kh - sh) + ph * 2) / sh;
    int ow = (w - (kw - sw) + pw * 2) / sw;
    auto ret = input->getDims();
    ret[input->getDims().size() - 1] = oh;
    ret[input->getDims().size() - 2] = ow;
    outputs[0]->setDims(ret);
    return ret;
}

double MaxPoolOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    cudnnPoolingDescriptor_t desc_op;

    curandGenerator_t gen;
    checkCurandError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurandError(
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock()));

    float *input, *output;
    cudnnTensorDescriptor_t desc_input, desc_output;
    int input_size = sizeof(float), output_size = sizeof(float);
    auto dims_input = inputs[0]->getDims();
    assert(dims_input.size() == 4);
    for (auto t : dims_input)
        input_size *= t;
    checkCudaError(cudaMalloc((void **)&input, input_size));
    checkCudnnError(cudnnCreateTensorDescriptor(&desc_input));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        desc_input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims_input[0],
        dims_input[1], dims_input[2], dims_input[3]));

    auto dims_output = outputs[0]->getDims();
    for (auto t : dims_output)
        output_size *= t;
    checkCudaError(cudaMalloc((void **)&output, output_size));
    checkCudnnError(cudnnCreateTensorDescriptor(&desc_output));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        desc_output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims_output[0],
        dims_output[1], dims_output[2], dims_output[3]));

    checkCurandError(
        curandGenerateUniform(gen, input, input_size / sizeof(float)));
    checkCudnnError(cudnnCreatePoolingDescriptor(&desc_op));
    checkCudnnError(cudnnSetPooling2dDescriptor(desc_op, CUDNN_POOLING_MAX,
                                                CUDNN_NOT_PROPAGATE_NAN, kh, kw,
                                                ph, pw, sh, sw));
    assert(dh == 1);
    assert(dw == 1);
    float alpha = 1, beta = 0;
    for (int i = 0; i < warmupRounds; ++i) {
        checkCudnnError(cudnnPoolingForward(pe->cudnnHandle(), desc_op, &alpha,
                                            desc_input, input, &beta,
                                            desc_output, output));
    }
    checkCudaError(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < rounds; ++i) {
        checkCudnnError(cudnnPoolingForward(pe->cudnnHandle(), desc_op, &alpha,
                                            desc_input, input, &beta,
                                            desc_output, output));
    }
    cudaEventRecord(stop);
    checkCudaError(cudaDeviceSynchronize());
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= rounds;
    PoolArgs args{kh, kw, ph, pw, sh, sw, dh, dw};
    pe->saveOpPerf(MaxPool, args, milliseconds);

    checkCudnnError(cudnnDestroyPoolingDescriptor(desc_op));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc_input));
    cudaFree(input);
    checkCudnnError(cudnnDestroyTensorDescriptor(desc_output));
    cudaFree(output);
    checkCurandError(curandDestroyGenerator(gen));
    return milliseconds;
}

AvgPoolOp::AvgPoolOp(Tensor *input, int kh, int kw, int ph, int pw, int sh,
                     int sw)
    : Operator(AvgPool, {input}, {}), kh(kh), kw(kw), ph(ph), pw(pw), sh(sh),
      sw(sw) {
    outputs.emplace_back(new Tensor());
    initHash();
    computeShape();
}

AvgPoolOp::AvgPoolOp(int kh, int kw, int ph, int pw, int sh, int sw)
    : Operator(AvgPool), kh(kh), kw(kw), ph(ph), pw(pw), sh(sh), sw(sw) {
    initHash();
}

void AvgPoolOp::initHash() {
    hash = type;
    hash = hashAppend(hash, kh);
    hash = hashAppend(hash, kw);
    hash = hashAppend(hash, ph);
    hash = hashAppend(hash, pw);
    hash = hashAppend(hash, sh);
    hash = hashAppend(hash, sw);
    hash = hashPack(hash);
}

Tensor *AvgPoolOp::compute() {
    // TODO: avgpool compute
    assert(false);
    return nullptr;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
AvgPoolOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    return {{DimRange::getAllPos()}, [this]() { return compute() != nullptr; }};
}

Dim AvgPoolOp::computeShape() {
    auto input = inputs[0];
    auto h = input->getDims()[input->getDims().size() - 1],
         w = input->getDims()[input->getDims().size() - 2];
    int oh = (h - (kh - sh) + ph * 2) / sh;
    int ow = (w - (kw - sw) + pw * 2) / sw;
    auto ret = input->getDims();
    ret[input->getDims().size() - 1] = oh;
    ret[input->getDims().size() - 2] = ow;
    outputs[0]->setDims(ret);
    return ret;
}

double AvgPoolOp::perf(PerfEngine *pe, int rounds, int warmupRounds) {
    cudnnPoolingDescriptor_t desc_op;

    curandGenerator_t gen;
    checkCurandError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCurandError(
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock()));

    float *input, *output;
    cudnnTensorDescriptor_t desc_input, desc_output;
    int input_size = sizeof(float), output_size = sizeof(float);
    auto dims_input = inputs[0]->getDims();
    assert(dims_input.size() == 4);
    for (auto t : dims_input)
        input_size *= t;
    checkCudaError(cudaMalloc((void **)&input, input_size));
    checkCudnnError(cudnnCreateTensorDescriptor(&desc_input));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        desc_input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims_input[0],
        dims_input[1], dims_input[2], dims_input[3]));

    auto dims_output = outputs[0]->getDims();
    for (auto t : dims_output)
        output_size *= t;
    checkCudaError(cudaMalloc((void **)&output, output_size));
    checkCudnnError(cudnnCreateTensorDescriptor(&desc_output));
    checkCudnnError(cudnnSetTensor4dDescriptor(
        desc_output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dims_output[0],
        dims_output[1], dims_output[2], dims_output[3]));

    checkCurandError(
        curandGenerateUniform(gen, input, input_size / sizeof(float)));
    checkCudnnError(cudnnCreatePoolingDescriptor(&desc_op));
    checkCudnnError(cudnnSetPooling2dDescriptor(
        desc_op, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN, kh, kw, ph, pw, sh, sw));
    float alpha = 1, beta = 0;
    for (int i = 0; i < warmupRounds; ++i) {
        checkCudnnError(cudnnPoolingForward(pe->cudnnHandle(), desc_op, &alpha,
                                            desc_input, input, &beta,
                                            desc_output, output));
    }
    checkCudaError(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < rounds; ++i) {
        checkCudnnError(cudnnPoolingForward(pe->cudnnHandle(), desc_op, &alpha,
                                            desc_input, input, &beta,
                                            desc_output, output));
    }
    cudaEventRecord(stop);
    checkCudaError(cudaDeviceSynchronize());
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= rounds;
    PoolArgs args{kh, kw, ph, pw, sh, sw, 1, 1};
    pe->saveOpPerf(AvgPool, args, milliseconds);

    checkCudnnError(cudnnDestroyPoolingDescriptor(desc_op));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc_input));
    cudaFree(input);
    checkCudnnError(cudnnDestroyTensorDescriptor(desc_output));
    cudaFree(output);
    checkCurandError(curandDestroyGenerator(gen));
    return milliseconds;
}

AddOp::AddOp(TensorVec inputs) : Operator(Add, inputs, {}) {
    assert(checkValid(inputs));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

AddOp::AddOp() : Operator(Add) { initHash(); }

void AddOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

Tensor *AddOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto output = outputs[0];
    output->dataMalloc();
    auto o_ptr = output->getDataPtr();
    for (size_t i = 0, iEnd = output->size(); i < iEnd; ++i)
        o_ptr[i] = 0;
    auto oSz = output->size();
    for (auto input : inputs) {
        auto iSz = input->size();
        auto i_ptr = input->getDataPtr();
#pragma omp parallel for
        for (size_t i = 0; i < oSz; ++i)
            o_ptr[i] += i_ptr[i % iSz];
    }
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
AddOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {std::vector<DimRange>(inputs.size(), DimRange::getEmpty()),
                []() { return true; }};
    if (!dr.isSinglePos())
        return {std::vector<DimRange>(inputs.size(), dr),
                [this]() { return compute() != nullptr; }};
    auto pos = dr.getBegin();
    return {std::vector<DimRange>(inputs.size(), dr), [this, pos]() {
                auto output = outputs[0];
                output->dataMalloc();
                VType ret = 0;
                for (auto input : inputs)
                    ret += input->getBroadcastData(pos);
                return output->setData(pos, ret);
            }};
}

bool AddOp::checkValid(const TensorVec &tensors) {
    if (tensors.size() < 1)
        return false;
    for (auto tensor : tensors) {
        assert(tensor != nullptr);
        if (tensor->getType() != tensors[0]->getType())
            return false;
    }
    Dim dmO;
    for (auto &&in : inputs)
        if (in->getDims().size() > dmO.size())
            dmO = in->getDims();
    for (size_t i = 0, iEnd = tensors.size(); i < iEnd; ++i) {
        auto tensor = tensors[i];
        auto dmI = tensor->getDims();
        if (dmI.size() > dmO.size() ||
            !std::equal(dmI.rbegin(), dmI.rend(), dmO.rbegin())) {
            dbg(dmI);
            dbg(dmO);
            return false;
        }
    }
    return true;
}

Dim AddOp::computeShape() {
    Dim ret;
    for (auto &&in : inputs)
        if (in->getDims().size() > ret.size())
            ret = in->getDims();
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

SubOp::SubOp(TensorVec inputs) : Operator(Sub, inputs, {}) {
    assert(checkValid(inputs));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

SubOp::SubOp() : Operator(Sub) { initHash(); }

void SubOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

Tensor *SubOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto output = outputs[0];
    output->dataMalloc();
    auto o_ptr = output->getDataPtr(), i0_ptr = inputs[0]->getDataPtr(),
         i1_ptr = inputs[1]->getDataPtr();
    auto oSz = output->size();
    auto i0Sz = inputs[0]->size();
    auto i1Sz = inputs[1]->size();
#pragma omp parallel for
    for (size_t i = 0; i < oSz; ++i)
        o_ptr[i] = i0_ptr[i % i0Sz] - i1_ptr[i % i1Sz];
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
SubOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {std::vector<DimRange>(inputs.size(), DimRange::getEmpty()),
                []() { return true; }};
    if (!dr.isSinglePos())
        return {std::vector<DimRange>(inputs.size(), dr),
                [this]() { return compute() != nullptr; }};
    auto pos = dr.getBegin();
    return {std::vector<DimRange>(inputs.size(), dr), [this, pos]() {
                auto output = outputs[0];
                output->dataMalloc();
                return output->setData(pos,
                                       inputs[0]->getBroadcastData(pos) -
                                           inputs[1]->getBroadcastData(pos));
            }};
}

bool SubOp::checkValid(const TensorVec &tensors) {
    if (tensors.size() != 2)
        return false;
    for (auto tensor : tensors) {
        assert(tensor != nullptr);
        if (tensor->getType() != tensors[0]->getType())
            return false;
    }
    auto dmO = inputs[0]->getDims().size() > inputs[1]->getDims().size()
                   ? inputs[0]->getDims()
                   : inputs[1]->getDims();
    for (size_t i = 0, iEnd = tensors.size(); i < iEnd; ++i) {
        auto tensor = tensors[i];
        auto dmI = tensor->getDims();
        if (dmI.size() > dmO.size() ||
            !std::equal(dmI.rbegin(), dmI.rend(), dmO.rbegin())) {
            return false;
        }
    }
    return true;
}

Dim SubOp::computeShape() {
    auto ret = inputs[0]->getDims().size() > inputs[1]->getDims().size()
                   ? inputs[0]->getDims()
                   : inputs[1]->getDims();
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

MulOp::MulOp(TensorVec inputs) : Operator(Mul, inputs, {}) {
    assert(checkValid(inputs));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

MulOp::MulOp() : Operator(Mul) { initHash(); }

void MulOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

Tensor *MulOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto output = outputs[0];
    output->dataMalloc();
    auto o_ptr = output->getDataPtr();
    for (size_t i = 0, iEnd = output->size(); i < iEnd; ++i)
        o_ptr[i] = 1;
    auto oSz = output->size();
    for (auto input : inputs) {
        auto iSz = input->size();
        auto i_ptr = input->getDataPtr();
#pragma omp parallel for
        for (size_t i = 0; i < oSz; ++i)
            o_ptr[i] *= i_ptr[i % iSz];
    }
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
MulOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {std::vector<DimRange>(inputs.size(), DimRange::getEmpty()),
                []() { return true; }};
    if (!dr.isSinglePos())
        return {std::vector<DimRange>(inputs.size(), dr),
                [this]() { return compute() != nullptr; }};
    auto pos = dr.getBegin();
    return {std::vector<DimRange>(inputs.size(), dr), [this, pos]() {
                auto output = outputs[0];
                output->dataMalloc();
                VType ret = 1;
                for (auto input : inputs)
                    ret *= input->getBroadcastData(pos);
                return output->setData(pos, ret);
            }};
}

bool MulOp::checkValid(const TensorVec &tensors) {
    if (tensors.size() < 1)
        return false;
    for (auto tensor : tensors) {
        assert(tensor != nullptr);
        if (tensor->getType() != tensors[0]->getType())
            return false;
    }
    Dim dmO;
    for (auto &&in : inputs)
        if (in->getDims().size() > dmO.size())
            dmO = in->getDims();
    for (size_t i = 0, iEnd = tensors.size(); i < iEnd; ++i) {
        auto tensor = tensors[i];
        auto dmI = tensor->getDims();
        if (dmI.size() > dmO.size() ||
            !std::equal(dmI.rbegin(), dmI.rend(), dmO.rbegin())) {
            return false;
        }
    }
    return true;
}

Dim MulOp::computeShape() {
    Dim ret;
    for (auto &&in : inputs)
        if (in->getDims().size() > ret.size())
            ret = in->getDims();
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

DivOp::DivOp(TensorVec inputs) : Operator(Div, inputs, {}) {
    assert(checkValid(inputs));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

DivOp::DivOp() : Operator(Div) { initHash(); }

void DivOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

Tensor *DivOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto output = outputs[0];
    output->dataMalloc();
    auto o_ptr = output->getDataPtr(), i0_ptr = inputs[0]->getDataPtr(),
         i1_ptr = inputs[1]->getDataPtr();
    auto oSz = output->size();
    auto i0Sz = inputs[0]->size();
    auto i1Sz = inputs[1]->size();
#pragma omp parallel for
    for (size_t i = 0; i < oSz; ++i)
        o_ptr[i] = i0_ptr[i % i0Sz] - i1_ptr[i % i1Sz];
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
DivOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {std::vector<DimRange>(inputs.size(), DimRange::getEmpty()),
                []() { return true; }};
    if (!dr.isSinglePos())
        return {std::vector<DimRange>(inputs.size(), dr),
                [this]() { return compute() != nullptr; }};
    auto pos = dr.getBegin();
    return {std::vector<DimRange>(inputs.size(), dr), [this, pos]() {
                auto output = outputs[0];
                output->dataMalloc();
                return output->setData(pos,
                                       inputs[0]->getBroadcastData(pos) /
                                           inputs[1]->getBroadcastData(pos));
            }};
}

bool DivOp::checkValid(const TensorVec &tensors) {
    if (tensors.size() != 2)
        return false;
    for (auto tensor : tensors) {
        assert(tensor != nullptr);
        if (tensor->getType() != tensors[0]->getType())
            return false;
    }
    auto dmO = inputs[0]->getDims().size() > inputs[1]->getDims().size()
                   ? inputs[0]->getDims()
                   : inputs[1]->getDims();
    for (size_t i = 0, iEnd = tensors.size(); i < iEnd; ++i) {
        auto tensor = tensors[i];
        auto dmI = tensor->getDims();
        if (dmI.size() > dmO.size() ||
            !std::equal(dmI.rbegin(), dmI.rend(), dmO.rbegin())) {
            return false;
        }
    }
    return true;
}

Dim DivOp::computeShape() {
    auto ret = inputs[0]->getDims().size() > inputs[1]->getDims().size()
                   ? inputs[0]->getDims()
                   : inputs[1]->getDims();
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

PowOp::PowOp(Tensor *input, int pow) : Operator(Pow, inputs, {}), pow(pow) {
    assert(checkValid(inputs));
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

PowOp::PowOp(int pow) : Operator(Pow), pow(pow) { initHash(); }

void PowOp::initHash() {
    hash = type;
    hash = hashAppend(hash, pow);
    hash = hashPack(hash);
}

Tensor *PowOp::compute() {
    if (outputs[0]->isComputed())
        return outputs[0];

    auto output = outputs[0];
    output->dataMalloc();
    auto o_ptr = output->getDataPtr(), i0_ptr = inputs[0]->getDataPtr();
    auto oSz = output->size();
#pragma omp parallel for
    for (size_t i = 0; i < oSz; ++i)
        o_ptr[i] = powVType(i0_ptr[i], pow);
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
PowOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {std::vector<DimRange>(inputs.size(), DimRange::getEmpty()),
                []() { return true; }};
    if (!dr.isSinglePos())
        return {std::vector<DimRange>(inputs.size(), dr),
                [this]() { return compute() != nullptr; }};
    auto pos = dr.getBegin();
    return {std::vector<DimRange>(inputs.size(), dr), [this, pos]() {
                auto output = outputs[0];
                output->dataMalloc();
                return output->setData(pos,
                                       powVType(inputs[0]->getData(pos), pow));
            }};
}

bool PowOp::checkValid(const TensorVec &tensors) {
    if (tensors.size() != 1)
        return false;
    for (auto tensor : tensors) {
        assert(tensor != nullptr);
        if (tensor->getType() != tensors[0]->getType())
            return false;
    }
    auto dm0 = tensors[0]->getDims();
    for (size_t i = 0, iEnd = tensors.size(); i < iEnd; ++i) {
        auto tensor = tensors[i];
        // tensor + scalar. inputs[0] must be tensor
        if (i != 0 && tensor->isScalar())
            continue;
        auto dmi = tensor->getDims();
        // TODO: broast add?
        if (dm0.size() != dmi.size())
            return false;
        for (size_t j = 0, jEnd = dm0.size(); j < jEnd; ++j)
            if (dm0[j] != dmi[j])
                return false;
    }
    return true;
}

Dim PowOp::computeShape() {
    auto ret = inputs[0]->getDims();
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

GatherOp::GatherOp(Tensor *data, Tensor *indices, int axis)
    : Operator(Gather, {data, indices}, {}), axis(axis) {
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

void GatherOp::initHash() {
    hash = type;
    hash = hashAppend(hash, axis);
    hash = hashPack(hash);
}

Tensor *GatherOp::compute() {
    assert(false);
    return nullptr;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
GatherOp::compute(DimRange dr) {
    assert(false);
    return {};
}

Dim GatherOp::computeShape() {
    auto dim = inputs[1]->getDims();
    for (size_t i = 0, iEnd = inputs[0]->getDims().size(); i < iEnd; i++) {
        if ((int)i != axis) {
            dim.push_back(inputs[0]->getDims()[i]);
        }
    }
    return dim;
}

ReduceMeanOp::ReduceMeanOp(Tensor *input, int axis)
    : Operator(ReduceMean, {input}, {}), axis(axis) {
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

void ReduceMeanOp::initHash() {
    hash = type;
    hash = hashAppend(hash, axis);
    hash = hashPack(hash);
}

Tensor *ReduceMeanOp::compute() {
    assert(false);
    return nullptr;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
ReduceMeanOp::compute(DimRange dr) {
    assert(false);
    return {};
}

Dim ReduceMeanOp::computeShape() {
    auto dim = inputs[0]->getDims();
    dim[axis] = 1;
    return dim;
}

SoftmaxOp::SoftmaxOp(Tensor *input, int axis)
    : Operator(Softmax, {input}, {}), axis(axis) {
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

void SoftmaxOp::initHash() {
    hash = type;
    hash = hashAppend(hash, axis);
    hash = hashPack(hash);
}

Tensor *SoftmaxOp::compute() {
    assert(false);
    return nullptr;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
SoftmaxOp::compute(DimRange dr) {
    assert(false);
    return {};
}

Dim SoftmaxOp::computeShape() {
    auto dim = inputs[0]->getDims();
    dim[axis] = 1;
    return dim;
}

IdentityOp::IdentityOp(Tensor *input) : Operator(Identity, {input}, {}) {
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

void IdentityOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

Tensor *IdentityOp::compute() {
    // TODO: fix bug
    // if (outputs[0]->isComputed())
    //     return outputs[0];

    auto input = inputs[0], output = outputs[0];
    output->dataMalloc();
    auto inputP = input->getDataPtr(), outputP = output->getDataPtr();
    for (size_t i = 0, iEnd = input->size(); i < iEnd; ++i)
        outputP[i] = inputP[i];
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
IdentityOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    return {{dr}, [this]() { return compute() != nullptr; }};
}

Dim IdentityOp::computeShape() {
    auto ret = inputs[0]->getDims();
    outputs[0]->setDims(ret);
    outputs[0]->setType(inputs[0]->getType());
    return ret;
}

Tensor *ReshapeOp::compute() {
    // if (outputs[0]->isComputed())
    //     return outputs[0];

    auto input = inputs[0], output = outputs[0];
    output->dataMalloc();
    auto inputP = input->getDataPtr(), outputP = output->getDataPtr();
    for (size_t i = 0, iEnd = input->size(); i < iEnd; ++i)
        outputP[i] = inputP[i];
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
ReshapeOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    if (dr.isEmpty())
        return {{DimRange::getEmpty()}, []() { return true; }};
    if (dr.isSinglePos()) {
        size_t offset = outputs[0]->getOffset(dr.getBegin());
        const Dim &input_dims = inputs[0]->getDims();
        Dim input_pos = input_dims;
        for (int i = (int)input_dims.size() - 1; i >= 0; --i) {
            input_pos[i] = offset % input_dims[i];
            offset /= input_dims[i];
        }
        return {{DimRange::getAllPos()},
                // return {{input_pos},
                [this]() { return compute() != nullptr; }};
    } else
        return {{DimRange::getAllPos()},
                [this]() { return compute() != nullptr; }};
}

void ReshapeOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

void ActivationOp::initHash() {
    hash = type;
    hash = hashAppend(hash, actType);
    hash = hashPack(hash);
}

// TODO FIXME: hash may be wrong. Should hash the TVM string.
void MemBoundOp::initHash() {
    hash = type;
    // TODO: change it to expr hash.
    hash = hashAppend(hash, exec_time);
    hash = hashPack(hash);
}

bool MemBoundOp::isComputeWeight() const {
    for (auto input : inputs) {
        if (input->getType() != Tensor::Weight) {
            return false;
        }
    }
    return true;
}

void MemBoundOp::setWeight() {
    if (isComputeWeight()) {
        for (auto output : outputs)
            output->setType(Tensor::Weight);
    }
}

Tensor *MemBoundOp::compute() {
    auto output = outputs[0];
    output->dataMalloc();
    std::unordered_map<std::string, nnet::Ref<std::vector<int>>> rangeInputs;
    for (int i = 0, iEnd = inputs.size(); i < iEnd; i++) {
        auto input = inputs[i];
        auto data = nnet::make_ref<std::vector<int>>(input->size());
        auto input_d = input->getDataPtr();
        for (int j = 0, jEnd = input->size(); j < jEnd; j++) {
            data->operator[](j) = input_d[j];
        }
        auto name = nnetInputs[i]->getName();
        rangeInputs.insert({name, data});
    }
    nnet::RangeOp range = nnet::as<nnet::RangeOpNode>(expr);
    const auto &rangeShape = range->getOutputShape();
    const auto &outputShape = output->getDims();
    // rangeShape and outputShape may extra dims of length 1.
    // But their sizes should be the same.
    assert((ssize_t)range->getOutputSize() == (ssize_t)output->size());
    const ssize_t iEnd = range->getOutputSize();
    /*#pragma omp parallel for default(none)\
        shared(range, output, rangeShape, outputShape, rangeInputs)  */
    for (ssize_t i = 0; i < iEnd; i++) {
        std::vector<int> rangePos(range->getNumOutputDims(), 0);
        std::vector<int> outputPos(outputShape.size(), 0);
        ssize_t t = i;
        for (int j = range->getNumOutputDims() - 1; 0 <= j; j--) {
            int extent = rangeShape[j];
            rangePos[j] = t % extent;
            t /= extent;
        }
        t = i;
        for (int j = outputShape.size() - 1; 0 <= j; j--) {
            int extent = outputShape[j];
            outputPos[j] = t % extent;
            t /= extent;
        }
        auto vals = nnet::Interpreter(rangeInputs).interpret(range, {rangePos});
        output->setData(outputPos, vals[0]);
    }
    output->setComputed();
    return output;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
MemBoundOp::compute(DimRange dr) {
    if (dr.notValid())
        return {};
    std::vector<DimRange> inDrs;
    if (dr.isEmpty()) {
        for (int i = 0, iEnd = inputs.size(); i < iEnd; i++) {
            inDrs.emplace_back(DimRange::getEmpty());
        }
        return {inDrs, []() { return true; }};
    }
    for (int i = 0, iEnd = inputs.size(); i < iEnd; i++) {
        inDrs.emplace_back(DimRange::getAllPos());
    }
    return {inDrs, [this, dr]() {
                auto output = outputs[0];
                std::unordered_map<std::string, nnet::Ref<std::vector<int>>>
                    rangeInputs;
                assert(inputs.size() == nnetInputs.size());
                for (int i = 0, iEnd = inputs.size(); i < iEnd; i++) {
                    auto input = inputs[i];
                    auto data = nnet::make_ref<std::vector<int>>(input->size());
                    auto input_d = input->getDataPtr();
                    for (int j = 0, jEnd = input->size(); j < jEnd; j++) {
                        data->operator[](j) = input_d[j];
                    }
                    auto name = nnetInputs[i]->getName();
                    rangeInputs.insert({name, data});
                }
                nnet::RangeOp range = nnet::as<nnet::RangeOpNode>(expr);
                auto &pos = dr.getBegin();
                auto vals =
                    nnet::Interpreter(rangeInputs).interpret(range, {pos});
                output->dataMalloc();
                return output->setData(pos, vals[0]);
            }};
}

ResizeOp::ResizeOp(Tensor *input, Tensor *sizes)
    : Operator(Resize, {input}, {}), sizes(sizes) {
    outputs.emplace_back(new Tensor());
    computeShape();
    initHash();
}

void ResizeOp::initHash() {
    hash = type;
    hash = hashPack(hash);
}

Tensor *ResizeOp::compute() {
    assert(false);
    return nullptr;
}

std::pair<std::vector<DimRange>, std::function<bool()>>
ResizeOp::compute(DimRange dr) {
    assert(false);
    return {};
}

Dim ResizeOp::computeShape() {
    auto dim = sizes->getDims();
    outputs[0]->setDims(dim);
    return dim;
}

void ConvOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Conv";
    auto input = inputs[0], weight = inputs[1];
    auto r = weight->getDims()[2];
    auto s = weight->getDims()[3];
    auto g = input->getDims()[1] / weight->getDims()[1];
    attr["dilations"] =
        "[" + std::to_string(dh) + "," + std::to_string(dw) + "]"; // "[1, 1]";
    attr["group"] = std::to_string(g);                             // "1";
    attr["kernel_shape"] =
        "[" + std::to_string(r) + "," + std::to_string(s) + "]"; // "[1, 1]";
    std::string pad = std::to_string(ph) + "," + std::to_string(pw);
    attr["pads"] = "[" + pad + "," + pad + "]"; // "[0, 0, 0, 0]";
    attr["strides"] =
        "[" + std::to_string(sh) + "," + std::to_string(sw) + "]"; // "[1, 1]";

    if (bias != nullptr) {
        Dim dim = bias->getDims();
        std::string name = "bias_" + std::to_string(getGuid());
        extra[name] = dim;
    }

    if (act == ActType::Relu) {
        attr["act"] = "Relu";
    } else {
        if (act == ActType::Sigmoid) {
            attr["act"] = "Sigmoid";
        }
    }
}

void MatmulOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "MatMul";

    if (bias != nullptr) {
        Dim dim = bias->getDims();
        std::string name = "bias_" + std::to_string(getGuid());
        extra[name] = dim;
    }
}

void ConvTransOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    // TODO
}

void G2BMMOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    // TODO
}

void GBMMLOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    // TODO
}

void PadOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Pad";
    attr["mode"] = "constant";

    std::vector<int> pads;
    for (auto x : begin)
        pads.emplace_back(x);
    for (auto x : end)
        pads.emplace_back(x);
    std::string name = "pads_" + std::to_string(getGuid());
    extra[name] = pads;
}

void SliceOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Slice";

    std::string guid_str = std::to_string(getGuid());
    extra["starts_" + guid_str] = begin;
    extra["ends_" + guid_str] = end;
}

void ConcatOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Concat";
    attr["axis"] = std::to_string(dim);
}

void SplitOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Split";
    attr["axis"] = std::to_string(dim);

    std::string name = "split_" + std::to_string(getGuid());
    extra[name] = sizes;
}

void TransposeOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    // TODO
}

void ExtendOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Extend";
    // TODO: attributes
}

void BatchNormOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "BatchNormalization";
    attr["epsilon"] = std::to_string(epsilon);
    attr["momentum"] = std::to_string(momentum);

    if (scale != nullptr) {
        Dim dim = scale->getDims();
        std::string name = "scale_" + std::to_string(getGuid());
        extra[name] = dim;
    }
    if (bias != nullptr) {
        Dim dim = bias->getDims();
        std::string name = "bias_" + std::to_string(getGuid());
        extra[name] = dim;
    }
    if (mean != nullptr) {
        Dim dim = mean->getDims();
        std::string name = "mean_" + std::to_string(getGuid());
        extra[name] = dim;
    }
    if (var != nullptr) {
        Dim dim = var->getDims();
        std::string name = "var_" + std::to_string(getGuid());
        extra[name] = dim;
    }
}

void MaxPoolOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "MaxPool";
    attr["auto_pad"] = "NOTSET";
    attr["dilations"] =
        "[" + std::to_string(dh) + "," + std::to_string(dw) + "]";
    attr["kernel_shape"] =
        "[" + std::to_string(kh) + "," + std::to_string(kw) + "]";
    std::string pad = std::to_string(ph) + "," + std::to_string(pw);
    attr["pads"] = "[" + pad + "," + pad + "]"; // "[0, 0, 0, 0]";
    attr["strides"] = "[" + std::to_string(sh) + "," + std::to_string(sw) + "]";
}

void AvgPoolOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "AveragePool";
    attr["auto_pad"] = "NOTSET";
    attr["count_include_pad"] = "0";
    attr["kernel_shape"] =
        "[" + std::to_string(kh) + "," + std::to_string(kw) + "]";
    std::string pad = std::to_string(ph) + "," + std::to_string(pw);
    attr["pads"] = "[" + pad + "," + pad + "]"; // "[0, 0, 0, 0]";
    attr["strides"] = "[" + std::to_string(sh) + "," + std::to_string(sw) + "]";
}

void AddOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Add";
}

void SubOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Sub";
}

void MulOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Mul";
}

void DivOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Div";
}

void PowOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Pow"; // TODO: add input as power if required
}

void GatherOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Gather";
    attr["axis"] = std::to_string(axis);
}

void ReduceMeanOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "ReduceMean";
    attr["axes"] = std::to_string(axis);
    attr["keepdims"] = "1";
}

void ReshapeOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Reshape";
    // TODO: compute or store the shape
}

void IdentityOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Identity";
}

void SoftmaxOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "Softmax";
    attr["axis"] = std::to_string(axis);
}

void ActivationOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    optype = "None";
    if (actType == Operator::ActType::Relu) {
        optype = "Relu";
    }
    if (actType == Operator::ActType::Sigmoid) {
        optype = "Sigmoid";
    }
}

void MemBoundOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    // TODO
}

void ResizeOp::getOptypeAttr(
    std::string &optype, std::map<std::string, std::string> &attr,
    std::map<std::string, std::vector<int>> &extra) const {
    // TODO
}

} // end of namespace tpm
