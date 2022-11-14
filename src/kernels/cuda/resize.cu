#include "cmath"
#include "cuda/cuda_common.h"
#include "cuda/resize.cuh"
#include <functional>

#ifndef GPU_LAMBDA
#define GPU_LAMBDA __device__
#endif

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
__device__ float half_pixel(int idx, float scale, int, int) {
    return (idx + 0.5) / scale - 0.5;
}

__device__ float pytorch_half_pixel(int idx, float scale, int length_resized,
                                    int) {
    return length_resized > 1 ? (idx + 0.5) / scale - 0.5 : 0;
}

__device__ float align_corners(int idx, float scale, int length_resized,
                               int length_original) {
    if (length_resized == 1)
        return 0;
    return (float)idx * (float)(length_original - 1) /
           (float)(length_resized - 1);
}

__device__ float asymmetric(int idx, float scale, int length_resized,
                            int length_original) {
    return idx / scale;
}
/*
__device__ float tf_crop_and_resize(int idx, float scale, int length_resized,
                            int length_original) {

}*/

// ATTENTION:The order of device functions in array must be consistent with the
// order in the enums of ResizeObj.
using nearest_mod_func_t = int (*)(float);
__device__ nearest_mod_func_t p_nearest_mode_fun[] = {
    round_prefer_floor, round_prefer_ceil, prefer_floor, prefer_ceil};

using coordinate_trans_mod_func_t = float (*)(int idxO, float scale, int lenO,
                                              int lenR);
__device__ coordinate_trans_mod_func_t p_cooridnate_trans_mode_func[] = {
    half_pixel, pytorch_half_pixel, align_corners, asymmetric};

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
            float scale = (float)metaData.oDims[i] / (float)metaData.inDims[i];
            int sIdx = nearestModeFun(transModeFun(
                dIdx, scale, metaData.oDims[i], metaData.inDims[i]));
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

// ATTENTION: Make sure dim <=4
typedef struct {
    int offset[16];
    float power[16];
} NeighborList;

int __device__ getLimitIdx(int idx, int limit) {
    if (idx < 0)
        return 0;
    if (idx > limit)
        return limit;
    return idx;
}

__global__ void _resize_kernel_linear(float *in, float *out, MetaData metaData,
                                      size_t num, int coordinateMode) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    while (tid < num) {
        auto dOffset = tid;
        auto neighborNum = 0;
        NeighborList neighborList;
        memset(&neighborList, 0, sizeof(neighborList));
        for (int i = metaData.nDims - 1; i >= 0; --i) {
            int dIdx = dOffset % metaData.oDims[i];
            float scale = metaData.scale[i];
            float sIdx = p_cooridnate_trans_mode_func[coordinateMode](
                dIdx, scale, scale * metaData.inDims[i], metaData.inDims[i]);

            int idx = std::floor(sIdx);
            float power = 1 - (sIdx - idx);

            // update neighborList
            if (metaData.inDims[i] == 1) {
                if (neighborNum == 0) {
                    neighborList.offset[0] = 0;
                    neighborList.power[0] = power;
                    neighborNum = 1;
                } else {
                    for (int j = 0; j < neighborNum; j++) {
                        neighborList.power[j] *= power;
                    }
                }
            } else {
                if (neighborNum == 0) {
                    neighborList.offset[0] =
                        getLimitIdx(idx, metaData.inDims[i] - 1) *
                        metaData.inStride[i];
                    neighborList.power[0] = power;
                    neighborList.offset[1] =
                        getLimitIdx(idx + 1, metaData.inDims[i] - 1) *
                        metaData.inStride[i];
                    neighborList.power[1] = 1 - power;
                    neighborNum = 2;
                } else {
                    for (int j = 0; j < neighborNum; j++) {
                        neighborList.offset[j + neighborNum] =
                            neighborList.offset[j] +
                            getLimitIdx(idx + 1, metaData.inDims[i] - 1) *
                                metaData.inStride[i];
                        neighborList.power[j + neighborNum] =
                            (neighborList.power[j]) * (1 - power);

                        neighborList.offset[j] +=
                            getLimitIdx(idx, metaData.inDims[i] - 1) *
                            metaData.inStride[i];
                        neighborList.power[j] *= power;
                    }
                    neighborNum *= 2;
                }
            }

            dOffset = dOffset / metaData.oDims[i];
        }

        float val = 0;
        for (int i = 0; i < neighborNum; ++i) {
            val += in[neighborList.offset[i]] * neighborList.power[i];
        }
        out[tid] = val;
        tid += stride;
    }
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
    _resize_kernel_nearest<<<blocksize, gridsize>>>(
        in, out, metaData, num, coordinateMode, nearestMode);
}

void resize_kernel_linear(float *in, float *out, const MetaData &metaData,
                          size_t num, int coordinateMode) {
    int blocksize = 32 * 16;
    auto gridsize = (num + blocksize - 1) / blocksize;
    IT_ASSERT(coordinateMode < sizeof(p_cooridnate_trans_mode_func) /
                                   sizeof(p_cooridnate_trans_mode_func[0]));
    _resize_kernel_linear<<<blocksize, gridsize>>>(in, out, metaData, num,
                                                   coordinateMode);
}
} // namespace infini
