#ifndef PERF_H
#define PERF_H

#include "common.h"
#include <tuple>

namespace tpm {

typedef std::tuple<int, // n
                   int, // c
                   int, // h
                   int, // w
                   int, // f
                   int, // r
                   int, // s
                   int, // ph
                   int, // pw
                   int, // sh
                   int, // sw
                   int, // dh
                   int, // dw
                   int, // g
                   int, // bias
                   int> // activation
    ConvArgs;

struct ConvResult {
    double time;
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspaceSize;
    bool fuseAct;
};

struct ConvTransResult {
    double time;
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t workspaceSize;
};

typedef std::tuple<bool, // transA
                   bool, // transB
                   int,  // b
                   int,  // m
                   int,  // n
                   int>  // k
    MatmulArgs;

typedef std::tuple<int, // b
                   int, // m
                   int, // k
                   int, // width
                   int> // dilation
    G2BMMGBMMLArgs;

typedef std::tuple<int, // kh
                   int, // kw
                   int, // ph
                   int, // pw
                   int, // sh
                   int, // sw
                   int, // dh, 1 for AvgPool
                   int> // dw, 1 for AvgPool
    PoolArgs;

struct MatmulResult {
    double time;
    bool useStrideBatchAPI;
    cublasGemmAlgo_t algo;
};

} // namespace tpm

#endif // PERF_H
