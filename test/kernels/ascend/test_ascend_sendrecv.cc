#ifdef INFINI_USE_HCCL
#include "ascend/ascend_runtime.h"
#include "ascend/hccl_communicator.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "operators/recv.h"
#include "operators/send.h"
#include "test.h"
#include <thread>

namespace infini {

void sendrecv(const string taskName, int deviceID, vector<float> data,
              const Shape &dataShape, int WORLD_SIZE, int source,
              int destination) {
    // Create Runtimes and initiate communication
    Runtime cpuRuntime = NativeCpuRuntimeObj::getInstance();
    Runtime ascnedRuntime = make_ref<ASCENDRuntimeObj>(deviceID);
    ascnedRuntime->initComm(taskName, WORLD_SIZE, deviceID);

    if (deviceID == source) {
        Graph gSend = make_ref<GraphObj>(ascnedRuntime);
        auto input = gSend->addTensor(Shape{static_cast<int>(data.size())},
                                      DataType::Float32);
        auto opSend =
            gSend->addOp<SendObj>(input, source, destination, nullptr);

        // Copy data from CPU to GPU
        gSend->dataMalloc();
        input->copyin(data);
        ascnedRuntime->run(gSend);
    }

    // ----------------

    if (deviceID == destination) {
        Graph gRecv = make_ref<GraphObj>(ascnedRuntime);
        int outputType = 1;
        // auto input =
        // gRecv->addTensor(Shape{static_cast<int>(data.size())},DataType::Float32);
        auto opRecv = gRecv->addOp<RecvObj>(nullptr, source, destination,
                                            dataShape, outputType, nullptr);
        gRecv->dataMalloc();
        ascnedRuntime->run(gRecv);

        auto result = opRecv->getOutput()->clone(cpuRuntime);
        EXPECT_TRUE(result->equalData(data));
    }
}

TEST(ASCEND_SendRecv1, run) {
    // Only 1 device gets data. Every rank should have the same data after
    // sendrecv.
    aclInit(nullptr);
    vector<float> data = {2., 3., 5., 6.};

    int WORLD_SIZE = 4;
    int source = 0;
    int destination = 2;
    std::vector<std::thread> threads;
    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
        threads.emplace_back(sendrecv, "test_sendrecv", gpu, data, Shape{2, 2},
                             WORLD_SIZE, source, destination);
    }

    for (auto &thread : threads) {
        thread.join();
    }
    aclFinalize();
}

// TEST(ASCEND_SendRecv2, run) {
//     // Only 1 device gets data. Every rank should have the same data after
//     // sendrecv.
//     vector<float> data = {2., 3., 5., 6.};//

//    int WORLD_SIZE = 3;
//    int source = 0;
//    int destination = 2;
//    std::vector<std::thread> threads;
//    for (int gpu = 0; gpu < WORLD_SIZE; ++gpu) {
//        threads.emplace_back(sendrecv, "test_sendrecv", gpu, data, Shape{2,
//        2},
//                             WORLD_SIZE, source, destination);
//    }//

//    for (auto &thread : threads) {
//        thread.join();
//    }
//}
} // namespace infini
#endif
