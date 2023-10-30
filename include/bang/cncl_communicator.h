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

  public:
    static Ref<CnclCommManager> getInstance(int worldSize);
    void syncAll();
    void check();
    void reset();
    static void resetInstance();
    // ~CnclCommManager();
};

class CnclCommunicatorObj final : public CommunicatorObj {
    //   private:
    //     cnclComm_t comm;
    //     cnrtQueue_t queue;
  private:
    int num_comms_total;

  public:
    CnclCommunicatorObj(const string &name, int worldSize, int rank)
        : CommunicatorObj(worldSize, rank) {
        // auto manager = CnclCommManager::getInstance(worldSize);
        // comm = manager->comms[rank];
        // queue = manager->queues[rank];
        num_comms_total = CnclCommManager::getInstance(worldSize)->num_comms;
    }

    // Get the actual cnclComm_t
    cnclComm_t getCnclComm() {
        return CnclCommManager::getInstance(worldSize)->comms[rank];
    }
    cnrtQueue_t getCnclQueue() {
        return CnclCommManager::getInstance(worldSize)->queues[rank];
    }

    // ~CnclCommunicatorObj() final {
    // CNCL_CHECK(cnclFreeComm(comm));
    // }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "CNCL communicator";
        return oss.str();
    }
};

} // namespace infini
