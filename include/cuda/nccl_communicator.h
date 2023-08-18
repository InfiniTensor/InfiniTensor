#pragma once
#include "core/communicator.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nccl.h>
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
    NcclCommunicatorObj(const string &name, int worldSize, int rank) {
        const std::string filePath("./" + name + "_nccl_id.bin");
        ncclUniqueId commId;
        if (rank == 0) {
            checkNcclError(ncclGetUniqueId(&commId));
            std::ofstream ofs(filePath, std::ios::binary);
            ofs.write((char *)&commId, sizeof(ncclUniqueId));

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
        if (rank == 0) {
            std::filesystem::remove(filePath);
        }
    }

    int getWorldSize() const override {
        int world_size;
        checkNcclError(ncclCommCount(comm, &world_size));
        return world_size;
    }

    virtual int getRank() const override {
        int rank;
        checkNcclError(ncclCommUserRank(comm, &rank));
        return rank;
    }

    virtual int getLocalRank() const override {
        int deviceID;
        checkNcclError(ncclCommCuDevice(comm, &deviceID));
        return deviceID;
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
