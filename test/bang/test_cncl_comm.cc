#ifdef INFINI_USE_CNCL
#include "bang/bang_runtime.h"
#include "bang/cncl_communicator.h"
#include "test.h"

static int WORLD_SIZE = 2;

namespace infini {

void allReduceSum(float *data, int deviceId) {
    // Create Runtime and setup communication
    BangRuntimeObj *bang_runtime = new BangRuntimeObj(deviceId);
    int rank = deviceId;
    bang_runtime->initComm("test_cncl_comm", WORLD_SIZE, rank);
    cnclComm_t comm =
        dynamic_cast<CnclCommunicatorObj &>(bang_runtime->getCommunicator())
            .getCnclComm();

    // Copy data
    float *data_mlu;
    checkBangError(bangMalloc(&data_mlu, sizeof(float)));
    checkBangError(
        bangMemcpy(data_mlu, data, sizeof(float), bangMemcpyHostToDevice));

    // Do AllReduce
    checkCnclError(
        cnclAllReduce(data_mlu, data_mlu, 1, cnclFloat, cnclSum, comm, 0));

    // Copy data back and sync device
    checkBangError(
        bangMemcpy(data, data_mlu, sizeof(float), bangMemcpyDeviceToHost));
    checkBangError(bangDeviceSynchronize());
}

// Setup communication between 2 threads, each controlling 1 MLU.
// Do AllReduce Sum on {1.0, 4.0}. Results should be {5.0, 5.0}.
TEST(CNCL, multi_mlu_communication) {
    int num_threads = WORLD_SIZE;
    float data[] = {1.0, 4.0};

    std::vector<std::thread> threads;
    for (int mlu = 0; mlu < num_threads; ++mlu) {
        threads.emplace_back(allReduceSum, &data[mlu], mlu);
    }
    for (auto &thread : threads) {
        thread.join();
    }

    for (int i = 0; i < num_threads; ++i) {
        ASSERT_EQ(data[i], 5.0f);
    }
}

} // namespace infini
#endif
