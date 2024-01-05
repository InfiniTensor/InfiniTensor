#ifdef INFINI_USE_CNCL
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/all_reduce.h"
#include "test.h"
#include <cncl.h>
#include <future>
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

    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            allReduce<AllReduceSumObj>("test_allreduce_sum", i, data[i], ans);
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

TEST(BANG_AllReduce, prod) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {10., 18.};

    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            allReduce<AllReduceProdObj>("test_allreduce_prod", i, data[i], ans);
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

TEST(BANG_AllReduce, min) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {2., 3.};

    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            allReduce<AllReduceMinObj>("test_allreduce_min", i, data[i], ans);
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

TEST(BANG_AllReduce, max) {
    vector<float> data[2] = {{2., 3.}, {5., 6.}};
    vector<float> ans = {5., 6.};

    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            allReduce<AllReduceMaxObj>("test_allreduce_max", i, data[i], ans);
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
