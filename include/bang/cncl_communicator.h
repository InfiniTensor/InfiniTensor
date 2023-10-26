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

class CnclCommManager {
  private:
    static std::map<std::string, CnclCommSet> comm_sets;

  public:
    static CnclCommSet getComms(std::string task_name, int worldSize) {
        if (comm_sets.find(task_name) == comm_sets.end()) {

        } else {
            return comm_sets.at(task_name);
        }
    }

}

class CnclCommSet {
  public:
    cnclComm_t *comms;
    cnrtQueue_t *queues;
    int *dev_list;
    int *rank_list;
    int num_comms;

  private:
    CnclCommManager(int worldSize);
    static Ref<CnclCommManager> instance;
    static std::mutex mutex;
    static std::once_flag flag;

  public:
    static Ref<CnclCommManager> getInstance(int worldSize);
    static void destroyInstance();
    ~CnclCommManager();
};

class CnclCommunicatorObj final : public CommunicatorObj {
  private:
    cnclComm_t comm;
    cnrtQueue_t queue;

  public:
    CnclCommunicatorObj(const string &name, int worldSize, int rank)
        : CommunicatorObj(worldSize, rank) {
        auto manager = CnclCommManager::getInstance(worldSize);
        comm = manager->comms[rank];
        queue = manager->queues[rank];
    }

    // Get the actual cnclComm_t
    cnclComm_t getCnclComm() { return comm; }
    cnrtQueue_t getCnclQueue() { return queue; }

    ~CnclCommunicatorObj() final {
        // CNCL_CHECK(cnclFreeComm(comm));
    }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "CNCL communicator";
        return oss.str();
    }
};

} // namespace infini
