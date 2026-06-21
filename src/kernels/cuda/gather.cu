#include "cuda/cuda_common.h"
#include "cuda/gather.h"

template <typename T>
__device__ T gatheredOffset2Offset(int gOffset,
                                   infini::GatherMetaData metaData) {
    T offset = 0;
    for (int i = metaData.inNDim - 1, k = metaData.outNDim - 1; i >= 0; --i) {
        T idx = 0;
        if (i == metaData.axis) {
            T idxOffset = 0;
            for (int j = metaData.idxNDim - 1; j >= 0; --j) {
                T p = gOffset % metaData.idxDim[j];
                gOffset = gOffset / metaData.idxDim[j];
                idxOffset += p * metaData.idxStride[j];
            }

            idx = static_cast<T *>(metaData.indexValue)[idxOffset];
            k = k - metaData.idxNDim;

        } else {
            idx = gOffset % metaData.outDim[k];
            gOffset = gOffset / metaData.outDim[k];
            --k;
        }
        offset += idx * metaData.inStride[i];
    }
    return offset;
}

template <typename dataT, typename T>
__global__ void _gather_kernel(dataT *in, dataT *out,
                               infini::GatherMetaData metaData, size_t num) {
    T tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num) {
        T offset = gatheredOffset2Offset<T>(tid, metaData);
        out[tid] = in[offset];
    }
}
template <typename T, typename Tind>
__global__ void blockGatherKernel(T const *input, Tind const *indices,
                                  T *output, int dimsize, int behindsize,
                                  int indsize, int numThread) {
    // input = [A, dimsize, D, E], indices = [B, C], axis = 1, output = [A, B,
    // C, D, E] frontsize表示axis前面的所有，即frontsize = A,
    // behindisize表示axis后面所有，即behindsize = DE, indsize = BC
    // 专门对付indsize * behindsize 比较大的情况，此时num_blocks_x = frontsize
    //  blockIdx.x解决frontsize， (threadIdx.x + blockIdx.y * blockDim.x) *
    //  numThread解决indsize * behindsize
    // blockDim.x = 1, numThread表示一个线程处理多少个元素
    int otherIdx = threadIdx.x + blockIdx.y * blockDim.x;
    int frontIdx = blockIdx.x;

    for (int i = 0; i < numThread; i++) {
        int idx = otherIdx * numThread +
                  i; // idx = indicesIdx * behindsize + behindIdx
        if (idx >= indsize * behindsize) {
            break;
        }
        int indicesIdx = idx / behindsize;
        int behindIdx = idx % behindsize;
        int inputIdx = frontIdx * dimsize * behindsize +
                       indices[indicesIdx] * behindsize + behindIdx;
        int outputIdx = frontIdx * indsize * behindsize + idx;
        output[outputIdx] = input[inputIdx];
    }
}
template <typename T, typename Tind>
__global__ void warpGatherKernel(T const *input, Tind const *indices, T *output,
                                 int frontsize, int dimsize, int behindsize,
                                 int indsize, int numThread) {
    // input = [A, dimsize, D, E], indices = [B, C], axis = 1, output = [A, B,
    // C, D, E] frontsize表示axis前面的所有，即frontsize = A,
    // behindisize表示axis后面所有，即behindsize = DE, indsize = BC
    // 专门对付dimsize * behindsize 比较大的情况，此时num_blocks_x = frontsize
    //  blockIdx.x * blockDim.x + threadIdx.x解决frontsize， (threadIdx.y +
    //  blockIdx.y * blockDim.y) * numThread解决indsize * behindsize
    // numThread表示一个线程处理多少个元素
    int otherIdx = threadIdx.y + blockIdx.y * blockDim.y;
    int frontIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (frontIdx >= frontsize) {
        return;
    }
    for (int i = 0; i < numThread; i++) {
        int idx = otherIdx * numThread +
                  i; // idx = indicesIdx * behindsize + behindIdx
        if (idx >= indsize * behindsize) {
            break;
        }
        int indicesIdx = idx / behindsize;
        int behindIdx = idx % behindsize;
        int inputIdx = frontIdx * dimsize * behindsize +
                       indices[indicesIdx] * behindsize + behindIdx;
        int outputIdx = frontIdx * indsize * behindsize + idx;
        output[outputIdx] = input[inputIdx];
    }
}

namespace infini {
template <typename T>
void gather_kernel(T *in, T *out, GatherMetaData metaData, size_t num) {
    int blockSize = 32 * 16;
    int gridSize = (num + blockSize - 1) / blockSize;
    if (metaData.indexType == DataType::Int64) {
        _gather_kernel<T, int64_t>
            <<<gridSize, blockSize, 0, CUDAStream::getCurrentStream()>>>(
                in, out, metaData, num);
    } else {
        _gather_kernel<T, int>
            <<<gridSize, blockSize, 0, CUDAStream::getCurrentStream()>>>(
                in, out, metaData, num);
    }
}
template void gather_kernel<float>(float *in, float *out,
                                   GatherMetaData metaData, size_t num);
template void gather_kernel<half>(half *in, half *out, GatherMetaData metaData,
                                  size_t num);
template void gather_kernel<int8_t>(int8_t *in, int8_t *out,
                                    GatherMetaData metaData, size_t num);
template <typename T, typename Tind>
void gatherLaunch(void const *input, void const *indices, void *output,
                  int frontsize, int dimsize, int behindsize, int indsize) {
    int othersize = indsize * behindsize;
    int numThread =
        2; // 一个线程在othersize中处理多少个元素，这个参数对性能影响很大，需要仔细调整
    int count = othersize / numThread; // 处理othersize需要的总线程数目
    if (count > 1024) {
        int BLOCK_DIM = 1024;
        int num_block_x = frontsize;
        int num_block_y =
            (othersize + BLOCK_DIM * numThread - 1) / (BLOCK_DIM * numThread);
        dim3 block_dim(BLOCK_DIM, 1, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        blockGatherKernel<T, Tind>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>(
                (T *)input, (Tind *)indices, (T *)output, dimsize, behindsize,
                indsize, numThread);
    } else if (count > 31) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) /
                          (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>(
                (T *)input, (Tind *)indices, (T *)output, frontsize, dimsize,
                behindsize, indsize, numThread);
    } else if (count > 15) {
        int BLOCK_DIM_x = 64;
        int BLOCK_DIM_y = 16;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) /
                          (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>(
                (T *)input, (Tind *)indices, (T *)output, frontsize, dimsize,
                behindsize, indsize, numThread);
    } else if (count > 7) {
        int BLOCK_DIM_x = 128;
        int BLOCK_DIM_y = 8;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) /
                          (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>(
                (T *)input, (Tind *)indices, (T *)output, frontsize, dimsize,
                behindsize, indsize, numThread);
    } else {
        int BLOCK_DIM_x = 256;
        int BLOCK_DIM_y = 4;
        int num_block_x = (frontsize + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        int num_block_y = (othersize + BLOCK_DIM_y * numThread - 1) /
                          (BLOCK_DIM_y * numThread);
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);

        warpGatherKernel<T, Tind>
            <<<grid_dim, block_dim, 0, CUDAStream::getCurrentStream()>>>(
                (T *)input, (Tind *)indices, (T *)output, frontsize, dimsize,
                behindsize, indsize, numThread);
    }
}
void gather_nv_f32(void const *input, void const *indices, void *output,
                   int frontsize, int dimsize, int behindsize, int indsize,
                   DataType indexType) {
    if (indexType == DataType::Int64) {
        gatherLaunch<float, int64_t>(input, indices, output, frontsize, dimsize,
                                     behindsize, indsize);
    } else if (indexType == DataType::Int32) {
        gatherLaunch<float, int>(input, indices, output, frontsize, dimsize,
                                 behindsize, indsize);
    }
}
void gather_nv_f16(void const *input, void const *indices, void *output,
                   int frontsize, int dimsize, int behindsize, int indsize,
                   DataType indexType) {
    if (indexType == DataType::Int64) {
        gatherLaunch<half, int64_t>(input, indices, output, frontsize, dimsize,
                                    behindsize, indsize);
    } else if (indexType == DataType::Int32) {
        gatherLaunch<half, int>(input, indices, output, frontsize, dimsize,
                                behindsize, indsize);
    }
}
void gather_nv_f8(void const *input, void const *indices, void *output,
                  int frontsize, int dimsize, int behindsize, int indsize,
                  DataType indexType) {
    if (indexType == DataType::Int64) {
        gatherLaunch<int8_t, int64_t>(input, indices, output, frontsize,
                                      dimsize, behindsize, indsize);
    } else if (indexType == DataType::Int32) {
        gatherLaunch<int8_t, int>(input, indices, output, frontsize, dimsize,
                                  behindsize, indsize);
    }
}
} // namespace infini
