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

class CnclCommunicatorObj final : public CommunicatorObj {
  private:
    cnclComm_t *comms;

  public:
    CnclCommunicatorObj(const string &name, int worldSize, int rank)
        : CommunicatorObj(worldSize, rank) {
        const std::string filePath("./" + name + "_cncl_id.bin");
        cnclCliqueId clique_id;
        if (rank == 0) {
            CNCL_CHECK(cnclGetCliqueId(&clique_id));
            std::ofstream ofs(filePath, std::ios::binary);
            ofs.write((char *)&clique_id, sizeof(cnclCliqueId));

        } else {
            auto begin = std::chrono::steady_clock::now();
            while (!std::filesystem::exists(filePath)) {
                auto now = std::chrono::steady_clock::now();
                _IT_ASSERT_2(now < begin + std::chrono::seconds(10),
                             "time limit (10s) exceeded.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::ifstream ifs(filePath, std::ios::binary);
            ifs.read((char *)&clique_id, sizeof(cnclCliqueId));
        }

        int num_comms = 1;
        int *dev_list = new int[num_comms];
        int *rank_list = new int[num_comms];
        comms = new cnclComm_t[num_comms];
        uint32_t num_dev = 0;
        checkBangError(cnrtGetDeviceCount(&num_dev));

        for (int i = 0; i < num_comms; i++) {
            rank_list[i] = rank;
            dev_list[i] = rank_list[i] % num_dev;
        }

        CNCL_CHECK(cnclInitComms(comms, num_comms, dev_list, rank_list,
                                 worldSize, &clique_id));

        if (rank == 0) {
            std::filesystem::remove(filePath);
        }

        delete[] dev_list;
        delete[] rank_list;
    }

    ~CnclCommunicatorObj() {
        CNCL_CHECK(cnclDestroyComms(comms, 1));
        delete[] comms;
    }

    // Get the actual cnclComm_t
    cnclComm_t getCnclComm() { return comms[0]; }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "CNCL communicator";
        return oss.str();
    }
};

} // namespace infini
