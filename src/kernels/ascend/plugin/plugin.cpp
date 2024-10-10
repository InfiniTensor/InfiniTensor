#include "../../../../include/ascend/ascend_plugin_sub_kernel.h"

constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_ALIGN = 32;

#include "kernel_operator.h"
using namespace AscendC;
template <typename T>
__aicore__ void pluginsub(GM_ADDR x, GM_ADDR output, int N, int C, int H,
                          int W) {
    int32_t TILE_NUM = H - 4;
    int32_t TILE_LENGTH = (W - 4);
    bool isbytealign = (TILE_LENGTH * sizeof(T) % BYTE_ALIGN) == 0;
    if (!isbytealign) {
        TILE_LENGTH = ((W - 4) * sizeof(T) +
                       (BYTE_ALIGN - (W - 4) * sizeof(T) % BYTE_ALIGN)) /
                      sizeof(T); // num + (32 - num % 32)
    }

    int32_t inputSize = N * C * H * W;
    int32_t outputSize = N * C * (H - 4) * (W - 4) * 16;
    int32_t BLOCK_LENGTH_IN = inputSize / BLOCK_NUM;   // 1x(C/8)x36x78
    int32_t BLOCK_LENGTH_OUT = outputSize / BLOCK_NUM; // 1x(C/8)x32x16x74

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue, outQueueTail;
    TBuf<QuePosition::VECCALC> tmpPattern;
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
    pipe.InitBuffer(outQueueTail, BUFFER_NUM, 32);
    pipe.InitBuffer(tmpPattern, 32);

    uint32_t tileBlock =
        C / BLOCK_NUM / BUFFER_NUM; // C=32时，tileBlock=2, C=64时，tileBlock=4
    uint32_t loopCount = TILE_NUM * BUFFER_NUM * tileBlock;
    for (uint32_t i = 0; i < loopCount; ++i) {
        int32_t bufferIdx = i / (TILE_NUM * tileBlock);
        int32_t tileBlockIdx = i % (TILE_NUM * tileBlock) / TILE_NUM;
        int32_t tileIdx = i % TILE_NUM;
        int32_t tileOffset =
            (bufferIdx * tileBlock * H + tileBlockIdx * H + tileIdx) * W;
        int32_t tileOffsetOut = (bufferIdx * tileBlock * TILE_NUM +
                                 tileBlockIdx * TILE_NUM + tileIdx) *
                                16 * (W - 4);
        for (uint32_t j = 0; j < 16; ++j) {
            // copy in
            LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
            LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
            if (j < 5) {
                DataCopy(xLocal, xGm[tileOffset + j], TILE_LENGTH);
            } else if (j < 8) {
                DataCopy(xLocal, xGm[tileOffset + (j - 4) * W + 4],
                         TILE_LENGTH);
            } else if (j < 13) {
                DataCopy(xLocal, xGm[tileOffset + W * 4 + (j - 8)],
                         TILE_LENGTH);
            } else if (j < 16) {
                DataCopy(xLocal, xGm[tileOffset + W * (j - 12)], TILE_LENGTH);
            }

            DataCopy(yLocal, xGm[tileOffset + (W * 2 + 2)], TILE_LENGTH);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);

            // compute
            xLocal = inQueueX.DeQue<T>();
            yLocal = inQueueY.DeQue<T>();
            LocalTensor<T> zLocal = outQueue.AllocTensor<T>();
            Sub(zLocal, yLocal, xLocal, W - 4);

            // copy out
            if (!isbytealign) {
                LocalTensor<uint16_t> bufPattern = tmpPattern.Get<uint16_t>();
                bufPattern.SetValue(0, 0b1111110000000000);
                bufPattern.SetValue(1, 0b0000001111111111);
                LocalTensor<T> tailLocal = outQueueTail.AllocTensor<T>();
                uint32_t mask = 128;
                uint64_t rsvdCnt = 0;
                PipeBarrier<PIPE_ALL>();
                AscendC::GatherMask(tailLocal, zLocal[TILE_LENGTH - 32],
                                    bufPattern, true, mask, {1, 1, 8, 8},
                                    rsvdCnt);
                outQueue.EnQue<T>(zLocal);
                outQueueTail.EnQue<T>(tailLocal);
                inQueueX.FreeTensor(xLocal);
                inQueueY.FreeTensor(yLocal);

                zLocal = outQueue.DeQue<T>();
                tailLocal = outQueueTail.DeQue<T>();
                int32_t copy_little_Length =
                    (BYTE_ALIGN -
                     (TILE_LENGTH * sizeof(T) - (W - 4) * sizeof(T))) /
                    sizeof(T);                                          // 10
                int32_t copy_big_Length = (W - 4) - copy_little_Length; // 64
                DataCopy(outGm[tileOffsetOut + j * (W - 4)], zLocal,
                         copy_big_Length);
                int32_t tail_offset = tileOffsetOut + j * (W - 4) + 58;
                DataCopy(outGm[tail_offset], tailLocal, BYTE_ALIGN / sizeof(T));
                outQueueTail.FreeTensor(tailLocal);
                outQueue.FreeTensor(zLocal);

            } else {
                outQueue.EnQue<T>(zLocal);
                inQueueX.FreeTensor(xLocal);
                inQueueY.FreeTensor(yLocal);
                zLocal = outQueue.DeQue<T>();
                DataCopy(outGm[tileOffsetOut + j * (W - 4)], zLocal,
                         TILE_LENGTH);
                outQueue.FreeTensor(zLocal);
            }
        }
    }
}

extern "C" __global__ __aicore__ void kernel_operator_float(GM_ADDR input,
                                                            GM_ADDR output,
                                                            int N, int C, int H,
                                                            int W) {
    pluginsub<float>(input, output, N, C, H, W);
}

extern "C" __global__ __aicore__ void kernel_operator_half(GM_ADDR input,
                                                           GM_ADDR output,
                                                           int N, int C, int H,
                                                           int W) {
    pluginsub<half>(input, output, N, C, H, W);
}

extern "C" void plugin_sub_kernel(void *input, void *output,
                                  PluginMetaData plugin_meta_data, void *stream,
                                  int dtype) {
    auto inputShape = plugin_meta_data.input_shape;
    int N = inputShape[0];
    int C = inputShape[1];
    int H = inputShape[2];
    int W = inputShape[3];

    switch (dtype) {
    case 0:
        kernel_operator_float<<<BLOCK_NUM, nullptr, stream>>>(input, output, N,
                                                              C, H, W);
        break;
    case 1:
        kernel_operator_half<<<BLOCK_NUM, nullptr, stream>>>(input, output, N,
                                                             C, H, W);
        break;
    default:
        break;
    }
    return;
}
