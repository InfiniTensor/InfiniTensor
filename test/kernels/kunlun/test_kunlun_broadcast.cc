#ifdef INFINI_USE_XCCL
#include "core/graph.h"
#include "core/runtime.h"
#include "kunlun/kunlun_runtime.h"
#include "operators/broadcast.h"
#include "test.h"
#include <thread>
#include <xpu/bkcl.h>

static int WORLD_SIZE = 2;
static int root = 0;

namespace infini {

void broadcast(const string taskName, int deviceID, vector<float> data,
               vector<float> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime kunlunRuntime = make_ref<KUNLUNRuntimeObj>(deviceID);
    kunlunRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(kunlunRuntime);
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
    kunlunRuntime->run(g);
    // Copy output from GPU to CPU
    auto result = op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(result->equalData(ans));
}

TEST(KUNLUN_Broadcast, run) {
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
