// #ifdef INFINI_USE_XCCL
#include "core/graph.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/all_reduce.h"
#include "test.h"
#include "xpu/bkcl.h"
#include <thread>

static int WORLD_SIZE = 2;

using namespace infini;

template <typename OperatorObj>
void allReduce(const string taskName, int deviceID, vector<float> data, vector<float> ans) {
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime kunlunRuntime = make_ref<KUNLUNRuntimeObj>(deviceID);
    kunlunRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    Graph g = make_ref<GraphObj>(kunlunRuntime);
    auto input = g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op = g->addOp<OperatorObj>(input, nullptr);
    g->dataMalloc();
    input->copyin(data);
    kunlunRuntime->run(g);
    auto result = op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(result->equalData(ans));
}

TEST(KUNLUN_AllReduce, sum) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {7., 9.};
    std::vector<std::thread> threads;
    for (int rank = 0; rank < WORLD_SIZE; ++rank) {
        threads.emplace_back(allReduce<AllReduceSumObj>, "test_allreduce_sum", rank, data[rank], ans);
    }
    for (auto &thread : threads){
        thread.join();
    }
}

TEST(KUNLUN_AllReduce, max) {
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

TEST(KUNLUN_AllReduce, min) {
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
