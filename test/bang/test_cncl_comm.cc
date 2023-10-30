#ifdef INFINI_USE_CNCL
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include "test.h"

static int WORLD_SIZE = 2;

namespace infini {

void allReduceSum(float *data, int deviceId) {
    // Create Runtime and setup communication
    BangRuntimeObj *bang_runtime = new BangRuntimeObj();
    int rank = deviceId;
    bang_runtime->initComm("test_cncl_comm", WORLD_SIZE, rank);
    cnclComm_t comm =
        dynamic_cast<CnclCommunicatorObj &>(bang_runtime->getCommunicator())
            .getCnclComm();
    cnrtQueue_t queue =
        dynamic_cast<CnclCommunicatorObj &>(bang_runtime->getCommunicator())
            .getCnclQueue();
    // Copy data
    float *data_mlu;
    checkBangError(cnrtMalloc((void **)&data_mlu, sizeof(float)));
    checkBangError(
        cnrtMemcpy(data_mlu, data, sizeof(float), cnrtMemcpyHostToDev));
    // Do AllReduce
    CNCL_CHECK(
        cnclAllReduce(data_mlu, data_mlu, 1, cnclFloat, cnclSum, comm, queue));

    checkBangError(cnrtQueueSync(queue));
    // Copy data back and sync device
    checkBangError(
        cnrtMemcpy(data, data_mlu, sizeof(float), cnrtMemcpyDevToHost));
}

// Setup communication between 2 threads, each controlling 1 MLU.
// Do AllReduce Sum on {1.0, 4.0}. Results should be {5.0, 5.0}.
TEST(CNCL, multi_mlu_communication) {
    int num_threads = WORLD_SIZE;
    float data[] = {1.0, 4.0};

    auto manager = CnclCommManager::getInstance(num_threads);

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < num_threads; ++mlu) {
        threads.emplace_back(allReduceSum, &data[mlu], mlu);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    manager->reset();

    for (int i = 0; i < num_threads; ++i) {
        ASSERT_EQ(data[i], 5.0f);
    }
}

} // namespace infini
#endif
