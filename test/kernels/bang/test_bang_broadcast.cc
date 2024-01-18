#ifdef INFINI_USE_CNCL
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/broadcast.h"
#include "test.h"
#include <cncl.h>
#include <thread>

static int WORLD_SIZE = 2;
static int root = 0;

namespace infini {

void broadcast(const string taskName, int deviceID, vector<float> data,
               vector<float> ans) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime bangRuntime = make_ref<BangRuntimeObj>(deviceID);
    bangRuntime->initComm(taskName, WORLD_SIZE, deviceID);
    // Create Graph and insert allReduce operation
    Graph g = make_ref<GraphObj>(bangRuntime);
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
    bangRuntime->run(g);
    // Copy output from GPU to CPU
    auto result = op->getOutput()->clone(cpuRuntime);

    EXPECT_TRUE(result->equalData(ans));
}

TEST(BANG_Broadcast, run) {
    // Only 1 device gets data. Every rank should have the same data after
    // broadcast.
    vector<float> data = {2., 3., 5., 6.};
    vector<float> ans = {2., 3., 5., 6.};

    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            broadcast("test_broadcast", i, data, ans);
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
