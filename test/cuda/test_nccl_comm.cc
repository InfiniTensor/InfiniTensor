#ifdef INFINI_USE_NCCL
#include "cuda/cuda_runtime.h"
#include "cuda/nccl_communicator.h"
#include "test.h"

static int WORLD_SIZE = 2;

namespace infini {

void allReduceSum(float *data, int deviceId) {
    // Create Runtime and setup communication
    CudaRuntimeObj *cuda_runtime = new CudaRuntimeObj(deviceId);
    int rank = deviceId;
    cuda_runtime->initComm("test_nccl_comm", WORLD_SIZE, rank);
    ncclComm_t comm =
        dynamic_cast<NcclCommunicatorObj &>(cuda_runtime->getCommunicator())
            .getNcclComm();

    // Copy data
    float *data_gpu;
    checkCudaError(cudaMalloc(&data_gpu, sizeof(float)));
    checkCudaError(
        cudaMemcpy(data_gpu, data, sizeof(float), cudaMemcpyHostToDevice));

    // Do AllReduce
    checkNcclError(
        ncclAllReduce(data_gpu, data_gpu, 1, ncclFloat, ncclSum, comm, 0));

    // Copy data back and sync device
    checkCudaError(
        cudaMemcpy(data, data_gpu, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaDeviceSynchronize());
}

// Setup communication between 2 threads, each controlling 1 GPU.
// Do AllReduce Sum on {1.0, 4.0}. Results should be {5.0, 5.0}.
TEST(NCCL, multi_gpu_communication) {
    int num_threads = WORLD_SIZE;
    float data[] = {1.0, 4.0};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < num_threads; ++gpu) {
        threads.emplace_back(allReduceSum, &data[gpu], gpu);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    for (int i = 0; i < num_threads; ++i) {
        ASSERT_EQ(data[i], 5.0f);
    }

}

} // namespace infini
#endif