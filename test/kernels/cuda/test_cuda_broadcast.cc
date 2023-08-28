#ifdef INFINI_USE_NCCL
#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/broadcast.h"
#include "test.h"
#include <nccl.h>
#include <thread>

static int WORLD_SIZE = 2;
static int root = 0;

namespace infini {

void broadcast(const string taskName, int deviceID, vector<float> data,
               vector<float> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime cudaRuntime = make_ref<CudaRuntimeObj>(deviceID);
    cudaRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op = g->addOp<BroadcastObj>(input, nullptr, root);
    // Copy data from CPU to GPU
    g->dataMalloc();
    // Only rank 0 has the data
    if (deviceID == root) {
        input->copyin(data);
    }
    // Run broadcast operation
    cudaRuntime->run(g);
    // Copy output from GPU to CPU
    auto result = op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(result->equalData(ans));
}

TEST(CUDA_Broadcast, run) {
    // Only 1 device gets data. Every rank should have the same data after
    // broadcast.
    vector<float> data = {2., 3., 5., 6.};
    vector<float> ans = {2., 3., 5., 6.};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(broadcast, "test_broadcast", gpu, data, ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}
} // namespace infini
#endif
