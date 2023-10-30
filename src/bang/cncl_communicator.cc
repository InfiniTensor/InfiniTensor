#include "bang/cncl_communicator.h"

namespace infini {

CnclCommManager::CnclCommManager(int worldSize) {
    num_comms = worldSize;
    dev_list = new int[num_comms];
    rank_list = new int[num_comms];
    comms = new cnclComm_t[num_comms];
    queues = new cnrtQueue_t[num_comms];
    uint32_t num_dev = 0;
    checkBangError(cnrtGetDeviceCount(&num_dev));

    for (int i = 0; i < num_comms; i++) {
        rank_list[i] = i; // comm's rank
        dev_list[i] = rank_list[i] % num_dev;
    }
    CNCL_CHECK(cnclInitComms(comms, num_comms, dev_list, rank_list, worldSize,
                             nullptr));
}

Ref<CnclCommManager> CnclCommManager::getInstance(int worldSize) {
    if (!instance) {
        instance =
            std::shared_ptr<CnclCommManager>(new CnclCommManager(worldSize));
    }
    return instance;
}

void CnclCommManager::reset() {
    CNCL_CHECK(cnclDestroyComms(comms, num_comms));
    for (int i = 0; i < num_comms; i++) {
        checkBangError(cnrtQueueDestroy(queues[i]));
    }
    delete[] queues;
    delete[] comms;
    delete[] dev_list;
    delete[] rank_list;
    comms = nullptr;
    queues = nullptr;
    dev_list = nullptr;
    rank_list = nullptr;
    instance->resetInstance();
}

void CnclCommManager::resetInstance() { instance = nullptr; }

Ref<CnclCommManager> CnclCommManager::instance = nullptr;

} // namespace infini
