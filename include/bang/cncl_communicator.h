#pragma once
#include "bang_common.h"
#include "core/communicator.h"
#include <chrono>
#include <cncl.h>
#include <cnrt.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

namespace infini {

class CnclCommSet {
  public:
    cnclComm_t *comms;
    cnrtQueue_t *queues;
    int *dev_list;
    int *rank_list;
    int num_comms;

  public:
    CnclCommSet(int worldSize);
    CnclCommSet() {}
    void reset();
};

class CnclCommManager {
  public:
    std::map<std::pair<std::string, int>, CnclCommSet> comm_sets;
    std::map<std::pair<std::string, int>, int> alive_comms;

  private:
    CnclCommManager() {}
    static Ref<CnclCommManager> instance;
    static std::mutex mutex;

  public:
    static Ref<CnclCommManager> getInstance();
    CnclCommSet getCommSet(const string &name, int worldSize);
    void destroyComm(const string &name, int worldSize);
};

class CnclCommunicatorObj final : public CommunicatorObj {
  private:
    string task_name;
    int num_comms_total;

  public:
    CnclCommunicatorObj(const string &name, int worldSize, int rank)
        : CommunicatorObj(worldSize, rank) {
        task_name = name;
        num_comms_total = CnclCommManager::getInstance()
                              ->getCommSet(task_name, worldSize)
                              .num_comms;
    }

    ~CnclCommunicatorObj() {
        CnclCommManager::getInstance()->destroyComm(task_name, worldSize);
    }

    // Get the actual cnclComm_t
    cnclComm_t getCnclComm() {
        return CnclCommManager::getInstance()
            ->getCommSet(task_name, worldSize)
            .comms[rank];
    }
    cnrtQueue_t getCnclQueue() {
        return CnclCommManager::getInstance()
            ->getCommSet(task_name, worldSize)
            .queues[rank];
    }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "CNCL communicator";
        return oss.str();
    }
};

} // namespace infini
