#ifdef INFINI_USE_CNCL
#include "core/graph.h"
#include "core/runtime.h"
#include "bang/bang_runtime.h"
#include "operators/all_reduce.h"
#include "test.h"
#include <cncl.h>
#include <thread>

static int WORLD_SIZE = 2;

namespace infini {

template <typename OperatorObj>
void allReduce(const string taskName, int deviceID, vector<float> data,
               vector<float> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime bangRuntime = make_ref<BangRuntimeObj>(deviceID);
    bangRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(bangRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op = g->addOp<OperatorObj>(input, nullptr);
    // Copy data from CPU to MLU
    g->dataMalloc();
    input->copyin(data);
    // Run operation
    bangRuntime->run(g);
    // Copy output from MLU to CPU
    auto result = op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(result->equalData(ans));
}

TEST(BANG_AllReduce, sum) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {7., 9.};

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < WORLD_SIZE; ++mlu) {
        threads.emplace_back(allReduce<AllReduceSumObj>, "test_allreduce_sum",
                             mlu, data[mlu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(BANG_AllReduce, prod) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {10., 18.};

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < WORLD_SIZE; ++mlu) {
        threads.emplace_back(allReduce<AllReduceProdObj>, "test_allreduce_prod",
                             mlu, data[mlu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(BANG_AllReduce, min) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {2., 3.};

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < WORLD_SIZE; ++mlu) {
        threads.emplace_back(allReduce<AllReduceMinObj>, "test_allreduce_min",
                             mlu, data[mlu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(BANG_AllReduce, max) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {5., 6.};

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < WORLD_SIZE; ++mlu) {
        threads.emplace_back(allReduce<AllReduceMaxObj>, "test_allreduce_max",
                             mlu, data[mlu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

TEST(BANG_AllReduce, avg) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {3.5, 4.5};

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < WORLD_SIZE; ++mlu) {
        threads.emplace_back(allReduce<AllReduceAvgObj>, "test_allreduce_avg",
                             mlu, data[mlu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

} // namespace infini
#endif