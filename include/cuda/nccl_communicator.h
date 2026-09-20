#pragma once
#include "core/communicator.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nccl.h>
#include <system_error>
#include <thread>

#define checkNcclError(call)                                                   \
    {                                                                          \
        auto err = call;                                                       \
        if (ncclSuccess != err) {                                              \
            fprintf(stderr, "NCCL error in %s:%i : %s.\n", __FILE__, __LINE__, \
                    ncclGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

namespace infini {

class NcclCommunicatorObj final : public CommunicatorObj {
  private:
    ncclComm_t comm;

  public:
    NcclCommunicatorObj(const string &name, int worldSize, int rank)
        : CommunicatorObj(worldSize, rank) {
        const std::string filePath("./" + name + "_nccl_id.bin");
        const std::string readyPrefix("./" + name + "_nccl_ready_");
        ncclUniqueId commId;
        if (rank == 0) {
            std::error_code ec;
            std::filesystem::remove(filePath, ec);
            for (int i = 0; i < worldSize; ++i)
                std::filesystem::remove(readyPrefix + std::to_string(i) + ".bin", ec);
            checkNcclError(ncclGetUniqueId(&commId));
            const std::string tempPath = filePath + ".tmp";
            std::ofstream ofs(tempPath, std::ios::binary);
            ofs.write((char *)&commId, sizeof(ncclUniqueId));
            ofs.close();
            std::filesystem::rename(tempPath, filePath);

        } else {
            auto begin = std::chrono::steady_clock::now();
            while (!std::filesystem::exists(filePath)) {
                auto now = std::chrono::steady_clock::now();
                _IT_ASSERT_2(now < begin + std::chrono::seconds(10),
                             "time limit (10s) exceeded.");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::ifstream ifs(filePath, std::ios::binary);
            ifs.read((char *)&commId, sizeof(ncclUniqueId));
        }
        checkNcclError(ncclCommInitRank(&comm, worldSize, commId, rank));
        {
            std::ofstream ready(readyPrefix + std::to_string(rank) + ".bin",
                                std::ios::binary);
            ready.put('1');
        }
        if (rank == 0) {
            auto begin = std::chrono::steady_clock::now();
            while (true) {
                bool allReady = true;
                for (int i = 0; i < worldSize; ++i) {
                    if (!std::filesystem::exists(readyPrefix +
                                                 std::to_string(i) + ".bin")) {
                        allReady = false;
                        break;
                    }
                }
                if (allReady)
                    break;
                IT_ASSERT(std::chrono::steady_clock::now() <
                              begin + std::chrono::seconds(30),
                          "time limit (30s) exceeded waiting for NCCL ranks");
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            std::error_code ec;
            std::filesystem::remove(filePath, ec);
            for (int i = 0; i < worldSize; ++i)
                std::filesystem::remove(readyPrefix + std::to_string(i) + ".bin", ec);
        }
    }

    // Get the actual ncclComm_t
    ncclComm_t getNcclComm() { return comm; }

    void finalize() { checkNcclError(ncclCommFinalize(comm)); }

    ~NcclCommunicatorObj() final {
        finalize();
        checkNcclError(ncclCommDestroy(comm));
    }

    virtual string toString() const final {
        std::ostringstream oss;
        oss << "NCCL communicator";
        return oss.str();
    }
};

} // namespace infini
