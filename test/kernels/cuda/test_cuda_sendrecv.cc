#ifdef INFINI_USE_NCCL
#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/sendrecv.h"
#include "test.h"
#include <nccl.h>
#include <thread>

static int WORLD_SIZE = 4;
static int source = 0;
static int destination = 1;

namespace infini {

void sendrecv(const string taskName, int deviceID, vector<float> data,
              vector<float> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime cudaRuntime = make_ref<CudaRuntimeObj>(deviceID);
    cudaRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op =
        g->addOp<SendRecvObj>(input, nullptr, source, destination, deviceID);
    // Copy data from CPU to GPU
    g->dataMalloc();
    // Only rank 0 has the data
    if (deviceID == source) {
        input->copyin(data);
    }
    // Run sendrecv operation
    cudaRuntime->run(g);
    // Copy output from GPU to CPU
    auto result = op->getOutput()->clone(cpuRuntime);

    if (deviceID == destination) {
        EXPECT_TRUE(result->equalData(ans));
    }
}

TEST(CUDA_SendRecv, run) {
    // Only 1 device gets data. Every rank should have the same data after
    // sendrecv.
    vector<float> data = {2., 3., 5., 6.};
    vector<float> ans = {2., 3., 5., 6.};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(sendrecv, "test_sendrecv", gpu, data, ans);
    }

    for (auto &thread : threads) {
        thread.join();
    }
}
} // namespace infini
#endif