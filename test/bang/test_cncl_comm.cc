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
    cnrtQueue_t queue = bang_runtime->getBangQueue();
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
    ASSERT_EQ(*data, 5.0f);
}

// Setup communication between 2 threads, each controlling 1 MLU.
// Do AllReduce Sum on {1.0, 4.0}. Results should be {5.0, 5.0}.
TEST(CNCL, multi_mlu_communication) {
    float data[] = {1.0, 4.0};

    for (int i = 0; i < WORLD_SIZE; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            allReduceSum(&data[i], i);
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
