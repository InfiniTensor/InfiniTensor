#include "../../../../include/ascend/ascend_plugin_sub_kernel.h"

constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t BUFFER_NUM = 4;
constexpr int32_t TILE_NUM = 62;
constexpr int32_t TILE_LENGTH = 1024;

#include "kernel_operator.h"
using namespace AscendC;
template <typename T>
__aicore__ void pluginsub(GM_ADDR x, GM_ADDR output, size_t inputSize,
                          size_t outputSize, int C) {
    int32_t BLOCK_LENGTH_IN = inputSize / BLOCK_NUM;   // 1x(C/8)x66x1028
    int32_t BLOCK_LENGTH_OUT = outputSize / BLOCK_NUM; // 1x(C/8)x62x1024x16

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<T> xGm, outGm;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) +
                            BLOCK_LENGTH_IN * GetBlockIdx(),
                        BLOCK_LENGTH_IN);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output) +
                              BLOCK_LENGTH_OUT * GetBlockIdx(),
                          BLOCK_LENGTH_OUT);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(T));

    uint32_t tilePipe =
        C / BLOCK_NUM / BUFFER_NUM; // C=32时，tilePipe=1, C=64时，tilePipe=2
    uint32_t loopCount = TILE_NUM * BUFFER_NUM * tilePipe;
    for (uint32_t i = 0; i < loopCount; ++i) {
        for (uint32_t j = 0; j < 16; ++j) {
            // copy in
            LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
            LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
            int32_t bufferIdx = i / (TILE_NUM * tilePipe);
            int32_t tilePipeIdx = i % (TILE_NUM * tilePipe) / TILE_NUM;
            int32_t tileIdx = i % TILE_NUM;
            int32_t tileOffset =
                (bufferIdx * tilePipe * 66 + tilePipeIdx * 66 + tileIdx) * 1028;
            if (j < 5) {
                DataCopy(xLocal, xGm[tileOffset + j], TILE_LENGTH);
            } else if (j < 8) {
                DataCopy(xLocal, xGm[tileOffset + (j - 4) * 1028 + 4],
                         TILE_LENGTH);
            } else if (j < 13) {
                DataCopy(xLocal, xGm[tileOffset + 1028 * 4 + (j - 8)],
                         TILE_LENGTH);
            } else if (j < 16) {
                DataCopy(xLocal, xGm[tileOffset + 1028 * (j - 12)],
                         TILE_LENGTH);
            }

            DataCopy(yLocal, xGm[tileOffset + 2058], TILE_LENGTH);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);

            // compute
            xLocal = inQueueX.DeQue<T>();
            yLocal = inQueueY.DeQue<T>();
            LocalTensor<T> zLocal = outQueue.AllocTensor<T>();
            Sub(zLocal, yLocal, xLocal, TILE_LENGTH);

            outQueue.EnQue<T>(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);

            // copy out
            zLocal = outQueue.DeQue<T>();
            int32_t tileOffsetOut = (bufferIdx * tilePipe * TILE_NUM +
                                     tilePipeIdx * TILE_NUM + tileIdx) *
                                    16 * 1024;
            DataCopy(outGm[tileOffsetOut + j * 1024], zLocal, TILE_LENGTH);
            outQueue.FreeTensor(zLocal);
        }
    }
}

extern "C" __global__ __aicore__ void
kernel_operator_float(GM_ADDR input, GM_ADDR output, size_t inputSize,
                      size_t outputSize, int C) {
    pluginsub<float>(input, output, inputSize, outputSize, C);
}

extern "C" __global__ __aicore__ void
kernel_operator_half(GM_ADDR input, GM_ADDR output, size_t inputSize,
                     size_t outputSize, int C) {
    pluginsub<half>(input, output, inputSize, outputSize, C);
}

extern "C" void plugin_sub_kernel(void *input, void *output,
                                  PluginMetaData plugin_meta_data, void *stream,
                                  int dtype) {
    size_t inputSize = plugin_meta_data.input_size;
    size_t outputSize = plugin_meta_data.output_size;

    int C = plugin_meta_data.input_shape[1]; // 32 or 64
    switch (dtype) {
    case 0:
        kernel_operator_float<<<BLOCK_NUM, nullptr, stream>>>(
            input, output, inputSize, outputSize, C);
        break;
    case 1:
        kernel_operator_half<<<BLOCK_NUM, nullptr, stream>>>(
            input, output, inputSize, outputSize, C);
        break;
    default:
        break;
    }
    return;
}
