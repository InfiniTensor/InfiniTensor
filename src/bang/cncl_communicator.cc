#include "bang/cncl_communicator.h"

namespace infini {

Ref<CnclCommManager> CnclCommManager::getInstance() {
    if (!instance) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!instance) {
            instance = std::shared_ptr<CnclCommManager>(new CnclCommManager());
        }
    }
    return instance;
}

CnclCommSet CnclCommManager::getCommSet(const string &name, int worldSize) {
    std::unique_lock<std::mutex> lock(mutex);
    std::string task_name = name;
    auto key = std::make_pair(task_name, worldSize);
    auto it = comm_sets.find(key);
    if (it != comm_sets.end()) {
        CnclCommSet &target = it->second;
        return target;
    } else {
        CnclCommSet new_comm_set = CnclCommSet(worldSize);
        comm_sets.insert(std::make_pair(key, new_comm_set));
        alive_comms[key] = worldSize;
        return new_comm_set;
    }
}

void CnclCommManager::destroyComm(const string &name, int worldSize) {
    std::unique_lock<std::mutex> lock(mutex);
    std::string task_name = name;
    auto key = std::make_pair(task_name, worldSize);
    auto it = alive_comms.find(key);
    if (it != alive_comms.end()) {
        alive_comms[key] -= 1;
        if (alive_comms[key] == 0) {
            comm_sets[key].reset();
            comm_sets.erase(key);
            alive_comms.erase(key);
        }
    }
    return;
}

CnclCommSet::CnclCommSet(int worldSize) {
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

void CnclCommSet::reset() {
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
}

Ref<CnclCommManager> CnclCommManager::instance = nullptr;
std::mutex CnclCommManager::mutex;

} // namespace infini
