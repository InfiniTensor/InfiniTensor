#ifdef INFINI_USE_HCCL
#include "ascend/ascend_runtime.h"
#include "ascend/hccl_communicator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/all_reduce.h"
#include "test.h"
#include <thread>

static int WORLD_SIZE = 2;

namespace infini {

template <typename OperatorObj>
void allReduce(const string taskName, int deviceID, vector<float> data,
               vector<float> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime ascendRuntime = make_ref<ASCENDRuntimeObj>(deviceID);
    ascendRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(ascendRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op = g->addOp<OperatorObj>(input, nullptr);
    // Copy data from CPU to GPU
    g->dataMalloc();
    input->copyin(data);
    // Run operation
    ascendRuntime->run(g);
    // Copy output from GPU to CPU
    auto result = op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(result->equalData(ans));
}

// TEST(ASCEND_AllReduce, sum) {
//     aclInit(nullptr);
//     vector<float> data[2] = {{2., 3.}, {5., 6.}};
//     vector<float> ans = {7., 9.};
//
//     std::vector<std::thread> threads;
//     for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
//         threads.emplace_back(allReduce<AllReduceSumObj>,
//         "test_allreduce_sum",
//                              gpu, data[gpu], ans);
//     }
//     for (auto &thread : threads) {
//         thread.join();
//     }
//     aclFinalize();
// }

// TEST(ASCEND_AllReduce, prod) {
//     aclInit(nullptr);
//     vector<float> data[2] = {{2., 3.}, {5., 6.}};
//     vector<float> ans = {10., 18.};
//
//     std::vector<std::thread> threads;
//     for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
//         threads.emplace_back(allReduce<AllReduceProdObj>,
//         "test_allreduce_prod",
//                              gpu, data[gpu], ans);
//     }
//     for (auto &thread : threads) {
//         thread.join();
//     }
//     aclFinalize();
// }

// TEST(ASCEND_AllReduce, min) {
//     aclInit(nullptr);
//     vector<float> data[2] = {{2., 3.}, {5., 6.}};
//     vector<float> ans = {2., 3.};
//
//     std::vector<std::thread> threads;
//     for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
//         threads.emplace_back(allReduce<AllReduceMinObj>,
//         "test_allreduce_min",
//                              gpu, data[gpu], ans);
//     }
//     for (auto &thread : threads) {
//         thread.join();
//     }
//     aclFinalize();
// }

TEST(ASCEND_AllReduce, max) {
    aclInit(nullptr);
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
    aclFinalize();
}

// TEST(ASCEND_AllReduce, avg) {
//     vector<float> data[2] = {{2., 3.}, {5., 6.}};
//     vector<float> ans = {3.5, 4.5};
//
//     std::vector<std::thread> threads;
//     for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
//         threads.emplace_back(allReduce<AllReduceAvgObj>,
//         "test_allreduce_avg",
//                              gpu, data[gpu], ans);
//     }
//     for (auto &thread : threads) {
//         thread.join();
//     }
// }

} // namespace infini
#endif