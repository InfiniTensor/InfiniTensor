#ifdef INFINI_USE_HCCL
#include "ascend/ascend_runtime.h"
#include "ascend/hccl_communicator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/all_gather.h"
#include "test.h"
#include <thread>

static int WORLD_SIZE = 2;

namespace infini {

void allGather(const string taskName, int deviceID, vector<float> data,
               vector<vector<float>> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime ascnedRuntime = make_ref<ASCENDRuntimeObj>(deviceID);
    ascnedRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(ascnedRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op = g->addOp<AllGatherObj>(input, std::nullopt, WORLD_SIZE);
    // Copy data from CPU to MLU
    g->dataMalloc();
    input->copyin(data);
    // Run operation
    ascnedRuntime->run(g);
    // Copy output from MLU to CPU
    for (int i = 0; i < WORLD_SIZE; ++i) {
        auto result = op->getOutputs()[i]->clone(cpuRuntime);
        EXPECT_TRUE(result->equalData(ans[i]));
    }
}

TEST(ASCEND_AllGather, run) {
    aclInit(nullptr);

    vector<float> data[4] = {{2., 3.}, {5., 6.}, {7., 8.}, {9., 10.}};
    vector<vector<float>> ans = {{2., 3.}, {5., 6.}, {7., 8.}, {9., 10.}};

    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(allGather, "test_all_gather", gpu, data[gpu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    aclFinalize();
}

} // namespace infini
#endif
