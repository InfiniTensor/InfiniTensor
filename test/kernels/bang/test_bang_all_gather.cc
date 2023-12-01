#ifdef INFINI_USE_CNCL
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/all_gather.h"
#include "test.h"
#include <cncl.h>
#include <thread>

static int WORLD_SIZE = 2;

namespace infini {

void allGather(const string taskName, int deviceID, vector<float> data,
               vector<vector<float>> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime bangRuntime = make_ref<BangRuntimeObj>();
    bangRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(bangRuntime);
    auto input =
        g->addTensor(Shape{static_cast<int>(data.size())}, DataType::Float32);
    auto op = g->addOp<AllGatherObj>(input, std::nullopt, WORLD_SIZE);
    // Copy data from CPU to MLU
    g->dataMalloc();
    input->copyin(data);
    // Run operation
    bangRuntime->run(g);
    // Copy output from MLU to CPU
    for (int i = 0; i < WORLD_SIZE; ++i) {
        auto result = op->getOutputs()[i]->clone(cpuRuntime);
        EXPECT_TRUE(result->equalData(ans[i]));
    }
}

TEST(BANG_AllGather, run) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<vector<float>> ans = {{2., 3.}, {5., 6.}};

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < WORLD_SIZE; ++mlu) {
        threads.emplace_back(allGather, "test_all_gather", mlu, data[mlu], ans);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}
} // namespace infini
#endif
