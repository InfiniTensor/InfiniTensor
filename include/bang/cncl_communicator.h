#pragma once
#include "core/communicator.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cncl.h>
#include <cnrt.h>
#include <thread>

#define checkCnclError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (CNCL_STATUS_SUCCESS != err) {                                              \
            fprintf(stderr, "Cncl error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    cnclGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

class CnclCommunicatorObj final : public CommunicatorObj {
  private:
    cnclComm_t comm;
    int* dev_list;
    int* rank_list;
  public:
    CnclCommunicatorObj(const string &name, int worldSize, int rank, int *dev_list, int *rank_list)
        : CommunicatorObj(worldSize, rank) {
        const std::string filePath("./" + name + "_cncl_id.bin");
        cnclCliqueId commId;
        if (rank == 0) {
            checkCnclError(cnclGetCliqueId(&commId));
            std::ofstream ofs(filePath, std::ios::binary);
            ofs.write((char *)&commId, sizeof(cnclCliqueId));

        } else {
            auto begin = std::chrono::steady_clock::now();
            while (!std::filesystem::exists(filePath)) {
                auto now = std::chrono::steady_clock::now();
                _IT_ASSERT_2(now < begin + std::chrono::seconds(10),
                             "time limit (10s) exceeded.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::ifstream ifs(filePath, std::ios::binary);
            ifs.read((char *)&commId, sizeof(cnclCliqueId));
        }
        int num_comms = worldsize;
        dev_list = new int[num_comms];
        rank_list = new int[num_comms];
        uint32_t num_dev = 0;
        cnrtGetDeviceCount(&num_dev);
        for (int i = 0; i < num_comms; i++) {
            rank_list[i] = i;  // comm's rank
            dev_list[i] = rank_list[i] % num_dev;
        }
        checkCnclError(cnclCommInitRank(&comm, num_comms, dev_list, rank_list, worldsize, commId));
        if (rank == 0) {
            std::filesystem::remove(filePath);
        }
    }

    // Get the actual cnclComm_t
    cnclComm_t getCnclComm() { return comm; }

    void finalize() { checkCnclError(cnclCommFinalize(comm)); }

    ~CnclCommunicatorObj() final {
        finalize();
        checkCnclError(cnclCommDestroy(comm));
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

