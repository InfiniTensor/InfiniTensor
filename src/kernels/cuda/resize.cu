#include "cmath"
#include "cuda/cuda_common.h"
#include "cuda/resize.cuh"
#include <functional>

// nearest mode
__device__ int round_prefer_ceil(float x) {
    return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}

__device__ int round_prefer_floor(float x) {
    return (x > 0.0) ? floor(x + 0.4) : ceil(x - 0.4);
}

__device__ int prefer_floor(float x) { return std::floor(x); }

__device__ int prefer_ceil(float x) { return std::ceil(x); }

//  coordinate transform mode
__device__ float half_pixel(int idx, MetaData metaData, int dim) {
    return (idx + 0.5) / metaData.scale[dim] - 0.5;
}

__device__ float pytorch_half_pixel(int idx, MetaData metaData, int dim) {
    float resizedLen = metaData.scale[dim] * metaData.inDims[dim];
    return resizedLen > 1 ? (idx + 0.5) / metaData.scale[dim] - 0.5 : 0;
}

__device__ float align_corners(int idx, MetaData metaData, int dim) {
    float resizedLen = metaData.scale[dim] * metaData.inDims[dim];
    if (resizedLen == 1)
        return 0;
    return (float)idx * (float)(metaData.inDims[dim] - 1) /
           (float)(resizedLen - 1);
}

__device__ float asymmetric(int idx, MetaData metaData, int dim) {
    return idx / metaData.scale[dim];
}

__device__ float tf_crop_and_resize(int idx, MetaData metaData, int dim) {
    int resizedLen = metaData.scale[dim] * metaData.inDims[dim];
    return resizedLen > 1
               ? metaData.roiS[dim] * (metaData.inDims[dim] - 1) +
                     idx * (metaData.roiE[dim] - metaData.roiS[dim]) *
                         (metaData.inDims[dim] - 1) / (resizedLen - 1)
               : 0.5 * (metaData.roiS[dim] + metaData.roiE[dim]) *
                     (metaData.inDims[dim] - 1);
}

// ATTENTION:The order of device functions in array must be consistent with the
// order in the enums of ResizeObj.
using nearest_mod_func_t = int (*)(float);
__device__ nearest_mod_func_t p_nearest_mode_fun[] = {
    round_prefer_floor, round_prefer_ceil, prefer_floor, prefer_ceil};

using coordinate_trans_mod_func_t = float (*)(int idxO, MetaData metaData,
                                              int dim);
__device__ coordinate_trans_mod_func_t p_cooridnate_trans_mode_func[] = {
    half_pixel, pytorch_half_pixel, align_corners, asymmetric,
    tf_crop_and_resize};

template <typename T1, typename T2>
__device__ int nearestCoordinateTrans(int dOffset, MetaData metaData,
                                      T1 transModeFun, T2 nearestModeFun) {
    int sOffset = 0;
    for (int i = metaData.nDims - 1; i >= 0; --i) {
        int dIdx = dOffset % metaData.oDims[i];
        dOffset = dOffset / metaData.oDims[i];

        if (metaData.inDims[i] == metaData.oDims[i])
            sOffset += dIdx * metaData.inStride[i];
        else {
            int sIdx = nearestModeFun(transModeFun(dIdx, metaData, i));

            if (sIdx > metaData.inDims[i] - 1)
                sIdx = metaData.inDims[i] - 1;
            else if (sIdx < 0)
                sIdx = 0;
            sOffset += sIdx * metaData.inStride[i];
        }
    }
    return sOffset;
}

__global__ void _resize_kernel_nearest(float *in, float *out, MetaData metaData,
                                       size_t num, int coordinateMode,
                                       int nearestMode) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    while (tid < num) {
        int offset = nearestCoordinateTrans(
            tid, metaData, p_cooridnate_trans_mode_func[coordinateMode],
            p_nearest_mode_fun[nearestMode]);
        out[tid] = in[offset];
        tid += stride;
    }
}

int __device__ getLimitIdx(int idx, int limit) {
    if (idx < 0)
        return 0;
    if (idx > limit)
        return limit;
    return idx;
}

template <int N>
__device__ void getEvenNeighbors(float idx, int limit, int *neighbors) {
    for (int i = 0; i < N; i++) {
        neighbors[i] = getLimitIdx(std::floor(idx) - N / 2 + 1 + i, limit);
    }
}

__device__ void getLinearCoef(float ratio, float *coeffs) {
    coeffs[0] = 1 - ratio;
    coeffs[1] = ratio;
}

__device__ void getCubicCoef(float ratio, float *coeffs) {
    float A = -0.75;
    coeffs[0] =
        ((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A;
    coeffs[1] = ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1;
    coeffs[2] =
        ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1;
    coeffs[3] = ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A) *
                    ((1 - ratio) + 1) -
                4 * A;
}

using get_coef_func_t = void (*)(float, float *);

// N is neighbor number at each dim
template <int N, int totalNeighborNum>
__device__ void _resize_kernel_coeff(float *in, float *out, MetaData metaData,
                                     size_t num,
                                     coordinate_trans_mod_func_t coTransFunc,
                                     get_coef_func_t getCoefFunc) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    while (tid < num) {
        auto dOffset = tid;
        auto neighborCnt = 1;
        int offsetList[totalNeighborNum], offsetListOld[totalNeighborNum];
        float powerList[totalNeighborNum], powerListOld[totalNeighborNum];

        for (size_t i = 0; i < totalNeighborNum; ++i) {
            offsetList[i] = 0;
            powerList[i] = 1;
        }
        for (int i = metaData.nDims - 1; i >= 0; --i) {
            int dIdx = dOffset % metaData.oDims[i];
            float sIdx = coTransFunc(dIdx, metaData, i);

            int idx = std::floor(sIdx);
            float power[N];
            int neighbors[N];
            getCoefFunc(sIdx - idx, power);
            getEvenNeighbors<N>(sIdx, metaData.inDims[i] - 1, neighbors);

            for (int n = 0; n < neighborCnt; ++n) {
                offsetListOld[n] = offsetList[n];
                powerListOld[n] = powerList[n];
            }
            for (int n = 0; n < N; ++n) {
                for (int idx = 0; idx < neighborCnt; ++idx) {
                    offsetList[idx + n * neighborCnt] =
                        offsetListOld[idx] +
                        neighbors[n] * metaData.inStride[i];

                    powerList[idx + n * neighborCnt] =
                        powerListOld[idx] * power[n];
                }
            }
            neighborCnt = neighborCnt * N;
            dOffset = dOffset / metaData.oDims[i];
        }

        float val = 0;
        for (int i = 0; i < neighborCnt; ++i) {
            val += in[offsetList[i]] * powerList[i];
        }
        out[tid] = val;
        tid += stride;
    }
}

__global__ void _resize_kernel_linear_coeff(float *in, float *out,
                                            MetaData metaData, size_t num,
                                            int coordinateMode) {
    _resize_kernel_coeff<2, 16>(in, out, metaData, num,
                                p_cooridnate_trans_mode_func[coordinateMode],
                                getLinearCoef);
}

__global__ void _resize_kernel_cubic_coeff(float *in, float *out,
                                           MetaData metaData, size_t num,
                                           int coordinateMode) {
    _resize_kernel_coeff<4, 256>(in, out, metaData, num,
                                 p_cooridnate_trans_mode_func[coordinateMode],
                                 getCubicCoef);
}

namespace infini {
void resize_kernel_nearest(float *in, float *out, const MetaData &metaData,
                           size_t num, int coordinateMode, int nearestMode) {
    int blocksize = 32 * 16;
    auto gridsize = (num + blocksize - 1) / blocksize;
    IT_ASSERT(coordinateMode < sizeof(p_cooridnate_trans_mode_func) /
                                   sizeof(p_cooridnate_trans_mode_func[0]));
    IT_ASSERT(nearestMode <
              sizeof(p_nearest_mode_fun) / sizeof(p_nearest_mode_fun[0]));
    _resize_kernel_nearest<<<gridsize, blocksize>>>(
        in, out, metaData, num, coordinateMode, nearestMode);
}

void resize_kernel_linear(float *in, float *out, const MetaData &metaData,
                          size_t num, int coordinateMode) {
    int blocksize = 32 * 16;
    auto gridsize = (num + blocksize - 1) / blocksize;
    IT_ASSERT(coordinateMode < sizeof(p_cooridnate_trans_mode_func) /
                                   sizeof(p_cooridnate_trans_mode_func[0]));
    _resize_kernel_linear_coeff<<<gridsize, blocksize>>>(in, out, metaData, num,
                                                         coordinateMode);
}

void resize_kernel_cubic(float *in, float *out, const MetaData &metaData,
                         size_t num, int coordinateMode) {
    int blocksize = 32 * 16;
    auto gridsize = (num + blocksize - 1) / blocksize;
    IT_ASSERT(coordinateMode < sizeof(p_cooridnate_trans_mode_func) /
                                   sizeof(p_cooridnate_trans_mode_func[0]));
    _resize_kernel_cubic_coeff<<<gridsize, blocksize>>>(in, out, metaData, num,
                                                        coordinateMode);
}
} // namespace infini
