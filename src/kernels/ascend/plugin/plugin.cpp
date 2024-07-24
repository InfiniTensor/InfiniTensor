#include "../../../../include/ascend/ascend_plugin_sub_kernel.h"

constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t BUFFER_NUM = 4;
constexpr int32_t TILE_NUM = 62;
constexpr int32_t TILE_LENGTH = 1024;

#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void pluginsub(GM_ADDR x, GM_ADDR output,
                                                size_t inputSize,
                                                size_t outputSize, int C) {
    int32_t BLOCK_LENGTH_IN = inputSize / BLOCK_NUM;   // 1x(C/8)x66x1028
    int32_t BLOCK_LENGTH_OUT = outputSize / BLOCK_NUM; // 1x(C/8)x62x1024x16

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<float> xGm, outGm;

    xGm.SetGlobalBuffer((__gm__ float *)x + BLOCK_LENGTH_IN * GetBlockIdx(),
                        BLOCK_LENGTH_IN);
    outGm.SetGlobalBuffer((__gm__ float *)output +
                              BLOCK_LENGTH_OUT * GetBlockIdx(),
                          BLOCK_LENGTH_OUT);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(float));

    uint32_t tilePipe =
        C / BLOCK_NUM / BUFFER_NUM; // C=32时，tilePipe=1, C=64时，tilePipe=2
    uint32_t loopCount = TILE_NUM * BUFFER_NUM * tilePipe;
    for (uint32_t i = 0; i < loopCount; ++i) {
        for (uint32_t j = 0; j < 16; ++j) {
            // copy in
            LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
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
            xLocal = inQueueX.DeQue<float>();
            yLocal = inQueueY.DeQue<float>();
            LocalTensor<float> zLocal = outQueue.AllocTensor<float>();
            Sub(zLocal, yLocal, xLocal, TILE_LENGTH);

            outQueue.EnQue<float>(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);

            // copy out
            zLocal = outQueue.DeQue<float>();
            int32_t tileOffsetOut = (bufferIdx * tilePipe * TILE_NUM +
                                     tilePipeIdx * TILE_NUM + tileIdx) *
                                    16 * 1024;
            DataCopy(outGm[tileOffsetOut + j * 1024], zLocal, TILE_LENGTH);
            outQueue.FreeTensor(zLocal);
        }
    }
}

void plugin_sub_kernel(float *input, float *output,
                       PluginMetaData plugin_meta_data, void *stream) {
    size_t inputSize = plugin_meta_data.input_size;
    size_t outputSize = plugin_meta_data.output_size;

    int C = plugin_meta_data.input_shape[1]; // 32 or 64

    // 调用Kernel
    pluginsub<<<BLOCK_NUM, nullptr, stream>>>(input, output, inputSize,
                                              outputSize, C);
    return;
}
