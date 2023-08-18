#ifdef INFINI_USE_NCCL
#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/all_gather.h"
#include "test.h"
#include <nccl.h>
#include <thread>

static int WORLD_SIZE = 2;

namespace infini {

void allGather(const string taskName, int deviceID, vector<float> data,
               vector<vector<float>> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime cudaRuntime = make_ref<CudaRuntimeObj>(deviceID);
    cudaRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op = g->addOp<AllGatherObj>(input, std::nullopt, WORLD_SIZE);
    // Copy data from CPU to GPU
    g->dataMalloc();
    input->copyin(data);
    // Run operation
    cudaRuntime->run(g);
    // Copy output from GPU to CPU
    for (int i = 0; i < WORLD_SIZE; ++i) {
        auto result = op->getOutputs()[i]->clone(cpuRuntime);
        EXPECT_TRUE(result->equalData(ans[i]));
    }
}

TEST(CUDA_AllGather, run) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<vector<float>> ans = {{2., 3.}, {5., 6.}};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(allGather, "test_all_gather", gpu, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}
} // namespace infini
#endif