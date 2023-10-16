#pragma once
#include "bang_common.h"
#include "core/communicator.h"
#include <chrono>
#include <cncl.h>
#include <cnrt.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

namespace infini {

class CnclCommunicatorObj final : public CommunicatorObj {
  private:
    cnclComm_t *comms;
    cnrtQueue_t *queues;
    int *dev_list;
    int *rank_list;
    int num_comms;

  public:
    CnclCommunicatorObj(const string &name, int worldSize, int rank,
                        int *dev_list, int *rank_list, cnclComm_t *comms,
                        cnrtQueue_t *queues)
        : CommunicatorObj(worldSize, rank) {
        const std::string filePath("./" + name + "_cncl_id.bin");

        num_comms = worldSize;
        dev_list = new int[num_comms];
        rank_list = new int[num_comms];
        comms = new cnclComm_t[num_comms];
        queues = new cnrtQueue_t[num_comms];
        uint32_t num_dev = 0;
        cnrtGetDeviceCount(&num_dev);

        rank_list[rank] = rank; // comm's rank
        dev_list[rank] = rank_list[rank] % num_dev;
        checkBangError(cnrtQueueCreate(&queues[rank]));
        CNCL_CHECK(cnclInitComms(comms, num_comms, dev_list, rank_list,
                                 worldSize, nullptr));
        if (rank == 0) {
            std::filesystem::remove(filePath);
        }
    }

    // Get the actual cnclComm_t
    cnclComm_t getCnclComm() { return comms[rank]; }
    cnrtQueue_t getCnclQueue() { return queues[rank]; }

    ~CnclCommunicatorObj() final {

        CNCL_CHECK(cnclDestroyComms(comms, num_comms));
        checkBangError(cnrtQueueDestroy(queues[rank]));
        delete[] dev_list;
        delete[] rank_list;
    }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "CNCL communicator";
        return oss.str();
    }
};

} // namespace infini
