// #include "cuda/cuda_common.h"
// #include <cub/cub.cuh>
// template <int BLOCK_DIM>
// __launch_bounds__(BLOCK_DIM) __global__
//     void _dynamicQuantizeLinearKernel(float *input, float *outputY,
//                                       uint8_t yScale, uint8_t yZeroPoint,
//                                       int size) {
//     int i = threadIdx.x + blockIdx.x * BLOCK_DIM;
//     float maxData = __FLT_MAX__;
//     float minData = -__FLT_MAX__;
//     int remain = size % BLOCK_DIM;
//     int step = (size - remain) / BLOCK_DIM + 1;

//     if (threadIdx.x < remain) {
//         for (int ind = 0; ind < step; ind++) {

//             maxData = max(maxData, input[threadIdx.x * step + ind]);
//         }
//     } else {
//         for (int ind = 0; ind < step - 1; ind++) {

//             maxData =
//                 max(maxData, input[remain * step +
//                                    (threadIdx.x - remain) * (step - 1) +
//                                    ind]);
//         }
//     }
//     if (threadIdx.x < remain) {
//         for (int ind = 0; ind < step; ind++) {

//             minData = min(minData, input[threadIdx.x * step + ind]);
//         }
//     } else {
//         for (int ind = 0; ind < step - 1; ind++) {

//             minData =
//                 min(minData, input[remain * step +
//                                    (threadIdx.x - remain) * (step - 1) +
//                                    ind]);
//         }
//     }
//     typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;
//     __shared__ float maxTotal;
//     float blockMax = BlockReduce(temp_storage).Reduce(maxData, cub::Max());

//     __shared__ float minTotal;
//     float blockMin = BlockReduce(temp_storage).Reduce(minData, cub::Min());
//     if (threadIdx.x == 0) {
//         maxTotal = blockMax;
//         minTotal = blockMin;
//     }
//     __syncthreads();
//     int qmax = 255;
//     int qmin = 0;
//     yScale = (max(0, maxTotal) - min(0, minTotal)) / (qmax - qmin);
//     intermediate_zero_point = qmin - minTotal / yScale;
//     yZeroPoint = cast(round(saturate(itermediate_zero_point)));
//     if (i < size) {
//         outputY[i] = saturate(round(input[i] / yScale) + yZeroPoint);
//     }
// }
// //----------

// template <int BLOCK_DIM, int numPerThread>
// __launch_bounds__(BLOCK_DIM) __global__
//     void _dynamicQuantizeLinearKernel(float *input, float *outputY,
//                                       uint8_t yScale, uint8_t yZeroPoint,
//                                       int size) {
//     int i = threadIdx.x + blockIdx.x * BLOCK_DIM;
//     float maxData = __FLT_MAX__;
//     float minData = -__FLT_MAX__;
//     int remain = size % BLOCK_DIM;
//     int step = (size - remain) / BLOCK_DIM + 1;
//     float dataPerThread[numPerThread];
//     if (threadIdx.x < remain) {
//         for (int ind = 0; ind < step; ind++) {
//             dataPerThread[ind] = input[threadIdx.x * step + ind];
//             maxData = max(maxData, dataPerThread[ind]);
//         }
//     } else {
//         for (int ind = 0; ind < step - 1; ind++) {
//             dataPerThread[ind] =
//                 input[remain * step + (threadIdx.x - remain) * (step - 1) +
//                       ind];
//             maxData = max(maxData, dataPerThread[ind]);
//         }
//     }
//     if (threadIdx.x < remain) {
//         for (int ind = 0; ind < step; ind++) {

//             minData = min(minData, dataPerThread[ind]);
//         }
//     } else {
//         for (int ind = 0; ind < step - 1; ind++) {

//             minData = min(minData, dataPerThread[ind]);
//         }
//     }
//     typedef cub::BlockReduce<float, BLOCK_DIM> BlockReduce;
//     __shared__ typename BlockReduce::TempStorage temp_storage;
//     __shared__ float maxTotal;
//     float blockMax = BlockReduce(temp_storage).Reduce(maxData, cub::Max());

//     __shared__ float minTotal;
//     float blockMin = BlockReduce(temp_storage).Reduce(minData, cub::Min());
//     if (threadIdx.x == 0) {
//         maxTotal = blockMax;
//         minTotal = blockMin;
//     }
//     __syncthreads();
//     int qmax = 255;
//     int qmin = 0;
//     yScale = (max(0.0, maxTotal) - min(0.0, minTotal)) / (qmax - qmin);
//     intermediate_zero_point = qmin - minTotal / yScale;
//     yZeroPoint = cast(round(saturate(itermediate_zero_point)));
//     if (i < size) {
//         outputY[i] = saturate(round(input[i] / yScale) + yZeroPoint);
//     }
// }

// namespace infini {
// void dynamicQuantizeLinearKernel(float *input, float *outputY, uint8_t
// yScale,
//                                  uint8_t yZeroPoint, int size) {

//     if (size > 1024 * 128) {

//         int BLOCK_DIM = 1024;
//         int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
//         _dynamicQuantizeLinearKernel<1024><<<num_blocks, BLOCK_DIM>>>(
//             input, outputY, yScale, yZeroPoint, size);
//     } else if (size > 1024 * 64) {

//         int BLOCK_DIM = 1024;
//         int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
//         _dynamicQuantizeLinearKernel<1024, 128><<<num_blocks, BLOCK_DIM>>>(
//             input, outputY, yScale, yZeroPoint, size);
//     } else if (size > 1024 * 32) {

//         int BLOCK_DIM = 1024;
//         int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
//         _dynamicQuantizeLinearKernel<1024, 64><<<num_blocks, BLOCK_DIM>>>(
//             input, outputY, yScale, yZeroPoint, size);
//     } else if (size > 1024 * 16) {

//         int BLOCK_DIM = 1024;
//         int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
//         _dynamicQuantizeLinearKernel<1024, 32><<<num_blocks, BLOCK_DIM>>>(
//             input, outputY, yScale, yZeroPoint, size);
//     } else if (size > 1024 * 4) {

//         int BLOCK_DIM = 1024;
//         int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
//         _dynamicQuantizeLinearKernel<1024, 16><<<num_blocks, BLOCK_DIM>>>(
//             input, outputY, yScale, yZeroPoint, size);
//     } else if (size > 1024) {

//         int BLOCK_DIM = 1024;
//         int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
//         _dynamicQuantizeLinearKernel<1024, 4><<<num_blocks, BLOCK_DIM>>>(
//             input, outputY, yScale, yZeroPoint, size);
//     } else {
//         int BLOCK_DIM = 1024;
//         int num_blocks = (size + BLOCK_DIM - 1) / BLOCK_DIM;
//         _dynamicQuantizeLinearKernel<1024, 1><<<num_blocks, BLOCK_DIM>>>(
//             input, outputY, yScale, yZeroPoint, size);
//     }
// }
// } // namespace infini
