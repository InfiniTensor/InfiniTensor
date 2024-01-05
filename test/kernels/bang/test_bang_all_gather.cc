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
    Runtime bangRuntime = make_ref<BangRuntimeObj>(deviceID);
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

    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            allGather("test_all_gather", i, data[i], ans);
            exit(0); // Ensure child process exits to avoid unnecessary
                     // repetition in parent
        } else if (pid < 0) {
            std::cerr << "Error creating process" << std::endl;
        }
    }
    // Wait for all child processes to finish
    for (int i = 0; i < WORLD_SIZE; ++i) {
        wait(NULL);
    }
}

} // namespace infini
#endif
