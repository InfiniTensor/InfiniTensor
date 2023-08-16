#ifdef INFINI_USE_NCCL
#include "core/graph.h"
#include "core/runtime.h"
#include "cuda/cuda_runtime.h"
#include "cuda/cuda_utility.h"
#include "operators/all_reduce.h"
#include "test.h"
#include <nccl.h>
#include <thread>

static int WORLD_SIZE = 2;

namespace infini {

template <typename OperatorObj>
void allReduce(const string taskName, int deviceID, vector<float> data,
               vector<float> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime cudaRuntime = make_ref<CudaRuntimeObj>(deviceID);
    cudaRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(cudaRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto output =
        g->addTensor(Shape{static_cast<int>(ans.size())}, DataType::Float32);
    auto op = g->addOp<OperatorObj>(input, output);
    // Copy data from CPU to GPU
    g->dataMalloc();
    input->copyin(data);
    // Run operation
    cudaRuntime->run(g);
    // Copy output from GPU to CPU
    auto result = op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(result->equalData(ans));
}

TEST(CUDA_AllReduce, sum) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {7., 9.};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(allReduce<AllReduceSumObj>, "test_allreduce_sum",
                             gpu, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(CUDA_AllReduce, prod) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {10., 18.};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(allReduce<AllReduceProdObj>, "test_allreduce_prod",
                             gpu, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(CUDA_AllReduce, min) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {2., 3.};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(allReduce<AllReduceMinObj>, "test_allreduce_min",
                             gpu, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(CUDA_AllReduce, max) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {5., 6.};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(allReduce<AllReduceMaxObj>, "test_allreduce_max",
                             gpu, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(CUDA_AllReduce, avg) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {3.5, 4.5};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(allReduce<AllReduceAvgObj>, "test_allreduce_avg",
                             gpu, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

} // namespace infini
#endif