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
        // std::lock_guard<std::mutex> lock(mutex);
        // if (!instance) {
        instance =
            std::shared_ptr<CnclCommManager>(new CnclCommManager(worldSize));
        // }
    }
    // std::call_once(flag, [worldSize]() { instance = new
    // CnclCommManager(worldSize); });
    return instance;
}

void CnclCommManager::check() {
    printf("%p, %p, %p, %p\n", (void *)comms, (void *)queues, (void *)dev_list,
           (void *)rank_list);
}

void CnclCommManager::syncAll() {
    for (int i = 0; i < num_comms; i++) {
        checkBangError(cnrtQueueSync(queues[i]));
    }
}

void CnclCommManager::reset() {
    // for (int i = 0; i < num_comms; i++) {
    //     checkBangError(cnrtQueueSync(queues[i]));
    // }
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

// cnclComm_t *CnclCommManager::comms = nullptr;
// cnrtQueue_t *CnclCommManager::queues = nullptr;
// int *CnclCommManager::dev_list = nullptr;
// int *CnclCommManager::rank_list = nullptr;

Ref<CnclCommManager> CnclCommManager::instance = nullptr;
std::mutex CnclCommManager::mutex;

} // namespace infini
